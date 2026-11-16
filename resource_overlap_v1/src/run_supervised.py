"""One explicitly invoked, finite experiment. Never queues the next gate."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

P = Path.home() / 'fp4-llm'
O = P / 'results/resource_overlap_v1'


def gpu_state():
    r = subprocess.run(['sudo', '-n', '/usr/bin/nvidia-smi', '-q'], capture_output=True, text=True, timeout=10)
    if r.returncode:
        raise RuntimeError(r.stderr)
    pids = []
    for block in re.split(r'\n\s+Process ID\s*:', r.stdout)[1:]:
        if re.search(r'Type\s*:\s*C', block):
            pids.append(int(block.splitlines()[0].strip()))
    if 'GPU Recovery Action' in r.stdout and not re.search(r'GPU Recovery Action\s*:\s*None', r.stdout):
        raise RuntimeError('GPU recovery action is not None')
    return r.stdout, pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--target', choices=['decode', 'probe'], default='decode')
    ap.add_argument('--timeout', type=int, default=360)
    ap.add_argument('--profile', choices=['none', 'node', 'metrics'], default='none')
    ap.add_argument('options', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    assert re.fullmatch(r'[a-z0-9_]+', a.label)
    assert 1 <= a.timeout <= 900
    for d in ['raw', 'logs', 'sqlite', 'analysis', 'figures']:
        (O / d).mkdir(parents=True, exist_ok=True)
    dest = O / 'raw' / (a.label + '.json')
    assert not dest.exists() and not (O / 'logs' / (a.label + '.log')).exists(), 'Never overwrite a run'
    env = os.environ.copy()
    sourced = subprocess.check_output(['bash', '-c', 'source "$1"; env -0', 'overlap-env', str(P / 'env.sh')])
    env.update(dict(x.decode().split('=', 1) for x in sourced.split(b'\0') if b'=' in x))
    selected = dict(PRJ=str(P), HOME=str(Path.home()), TMPDIR=str(P / 'tmp'), TMP=str(P / 'tmp'), TEMP=str(P / 'tmp'),
                    TRITON_CACHE_DIR=str(P / 'cache/triton'), VLLM_CACHE_ROOT=str(P / 'cache/vllm'),
                    HF_HOME=str(P / 'cache/hf'), HF_HUB_OFFLINE='1', VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false',
                    VLLM_ENABLE_V1_MULTIPROCESSING='0', MAX_JOBS='2', XDG_CACHE_HOME=str(P / 'cache'),
                    FLASHINFER_WORKSPACE_BASE=str(P), FLASHINFER_AUTOTUNE_CACHE_DIR=str(P / 'cache/flashinfer_autotune'),
                    TORCHINDUCTOR_CACHE_DIR=str(P / 'cache/torchinductor'), TORCH_EXTENSIONS_DIR=str(P / 'cache/torch_extensions'),
                    CUDA_CACHE_PATH=str(P / 'cache/cuda'), PATH='/usr/local/cuda-13.3/bin:' + env['PATH'])
    env.update(selected)
    q, pids = gpu_state()
    who = subprocess.run(['/usr/local/bin/gpu-who'], capture_output=True, text=True, timeout=10)
    ps = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,uid=,comm=,args='], text=True)
    if pids:
        raise RuntimeError('GPU_BUSY_NO_RUN ' + str(pids))
    if any('wait_and_launch_live_controller' in line and 'python' in line for line in ps.splitlines()):
        raise RuntimeError('AUTO_LAUNCH_CONTROLLER_PRESENT_NO_RUN')
    opts = a.options[1:] if a.options[:1] == ['--'] else a.options
    script = O / 'src' / ('overlap_runner.py' if a.target == 'decode' else 'probe_check.py')
    cmd = [str(P / 'vllm/.venv/bin/python'), str(script), *opts, '--output', str(dest)]
    if a.profile != 'none':
        assert a.target == 'decode'
        cmd += ['--profile']
        flags = ['--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace=node',
                 '--capture-range=cudaProfilerApi', '--capture-range-end=stop-shutdown', '--kill=none',
                 '--output=' + str(O / 'raw' / a.label)]
        if a.profile == 'metrics':
            flags += ['--gpu-metrics-devices=0', '--gpu-metrics-set=gb20x', '--gpu-metrics-frequency=10000']
        cmd = ['sudo', '-n', '--preserve-env=TMPDIR', '/usr/local/sbin/gpu-profile', 'nsys', 'profile', *flags,
               'env', *[f'{k}={v}' for k, v in selected.items()], *cmd]
    record = dict(start=datetime.datetime.now().astimezone().isoformat(), args=vars(a), command=cmd,
                  environment=selected, gpu_who=who.stdout,
                  source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (O / 'src').glob('*.py')})
    (O / 'logs' / (a.label + '_precheck.txt')).write_text(q + '\n' + who.stdout)
    with (O / 'commands.txt').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print('START', a.label, flush=True)
    status, telemetry, known = 'running', [], set()
    log = O / 'logs' / (a.label + '.log')
    with log.open('w') as f:
        child = subprocess.Popen(cmd, cwd=P, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        known.add(child.pid)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started > a.timeout:
                    raise RuntimeError('HARD_TIMEOUT')
                q, pids = gpu_state()
                tree = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,uid='], text=True)
                rows = [tuple(map(int, line.split())) for line in tree.splitlines() if len(line.split()) == 3]
                for _ in range(12):
                    for pid, ppid, uid in rows:
                        if ppid in known:
                            known.add(pid)
                telemetry.append(dict(monotonic_ns=time.perf_counter_ns(), elapsed=time.monotonic() - started,
                                      status=q, compute_pids=pids))
                other = [pid for pid in pids if pid not in known]
                if other:
                    raise RuntimeError('OTHER_COMPUTE_TASK_APPEARED ' + str(other))
                time.sleep(2)
            status = 'completed' if child.returncode == 0 else 'failed'
        except BaseException as e:
            status = str(e)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    q, pids = gpu_state()
    record.update(status=status, returncode=child.returncode, known_pids=sorted(known),
                  remaining_compute_pids=pids, end=datetime.datetime.now().astimezone().isoformat())
    (O / 'logs' / (a.label + '_supervision.json')).write_text(json.dumps(dict(record=record, telemetry=telemetry), indent=2))
    (O / 'logs' / (a.label + '_postcheck.txt')).write_text(q)
    lines = log.read_text().splitlines()
    concise = [line for line in lines if line.startswith(('OVERLAP_', 'PROBE_CHECK')) or 'Error' in line or 'Traceback' in line]
    print('END', status, child.returncode, 'GPU_COMPUTE', pids, flush=True)
    print('\n'.join(concise[-16:] or lines[-8:]), flush=True)
    if status != 'completed' or child.returncode:
        raise SystemExit(4)
    if a.profile != 'none':
        rep = O / 'raw' / (a.label + '.nsys-rep')
        assert rep.is_file() and rep.stat().st_size > 0, 'EMPTY_TRACE'
        export = ['nsys', 'export', '--type=sqlite', '--output=' + str(O / 'sqlite' / (a.label + '.sqlite')), str(rep)]
        r = subprocess.run(export, capture_output=True, text=True, timeout=90, env=env)
        (O / 'logs' / (a.label + '_export.log')).write_text(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError('EXPORT_FAILED ' + r.stderr)
        print('EXPORTED', a.label, flush=True)


if __name__ == '__main__':
    main()
