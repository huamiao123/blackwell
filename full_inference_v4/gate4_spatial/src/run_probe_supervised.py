"""Run a single authorized standalone acceptance job on the shared 5090."""

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time

from g4_common import SPATIAL, V4, load, save, sha
from review_gate import validate_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--window', required=True, type=Path)
    parser.add_argument('--mode', choices=['acceptance', 'qkv_calibration', 'qkv_replay_nvfp4', 'qkv_replay_bf16',
                                          'qkv_overlap_nvfp4', 'qkv_overlap_bf16'], default='acceptance')
    args = parser.parse_args()
    if not re.fullmatch(r'probe_[a-z0-9_]+', args.run_id):
        raise ValueError('invalid run ID')
    window = load(args.window)
    qkv_replay = args.mode.startswith('qkv_') and args.mode != 'qkv_calibration'
    qkv_overlap = args.mode.startswith('qkv_overlap_')
    task = ('standalone_qkv_overlap' if qkv_overlap else
            'standalone_qkv_replay' if qkv_replay else 'standalone_probe_acceptance')
    deadline = validate_window(window, task, time.time(), socket.gethostname())
    project = Path.home() / 'fp4-llm'
    if not SPATIAL.is_relative_to(project.resolve()):
        raise RuntimeError('shared RTX 5090 project directory required')

    sys.path.insert(0, str(V4 / 'src'))
    from run_supervised import controllers, gpu_state

    validation_dir = 'qkv_replay_validation' if qkv_replay else 'probe_gpu_validation'
    run_dir = SPATIAL / validation_dir / 'raw' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    permit_path = run_dir / 'permit.json'
    worker = Path(__file__).with_name('validate_qkv_replay.py' if qkv_replay else 'validate_probe_gpu.py')
    command = [str(project / 'vllm/.venv/bin/python'), str(worker), str(run_dir)]
    if qkv_replay:
        command.extend(['--precision', args.mode.rsplit('_', 1)[-1]])
        if qkv_overlap:
            command.append('--overlap')
    else:
        command.extend(['--mode', args.mode])
    gpu_uuid = window['gpu_uuid']
    deadline = min(deadline, time.time() + 90)
    report = dict(status='PRECHECK', start=dt.datetime.now().astimezone().isoformat(),
                  gpu_uuid=gpu_uuid, command=command, worker_sha256=sha(worker),
                  supervisor_sha256=sha(Path(__file__)),
                  authorization_reference=window['authorization_reference'], telemetry=[])
    child = None

    def snapshot():
        state, pids = gpu_state()
        processes = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,uid=,comm=,args='],
                                            text=True, timeout=5)
        if gpu_uuid not in state:
            raise RuntimeError('GPU UUID mismatch')
        if controllers(processes):
            raise RuntimeError('automatic GPU controller present')
        report['telemetry'].append(dict(time=dt.datetime.now().astimezone().isoformat(), compute_pids=pids,
                                        state=state))
        return set(pids)

    try:
        if snapshot():
            raise RuntimeError('GPU_BUSY_NO_RUN')
        nonce = secrets.token_hex(24)
        compiled = load(SPATIAL / 'analysis/offline_compile/compile_manifest.json')
        permit = dict(scope=task if qkv_replay else 'probe_only', nonce=nonce, supervisor_pid=os.getpid(), deadline=deadline,
                      gpu_uuid=gpu_uuid, worker_sha256=sha(worker), cubin_sha256=compiled['cubin_sha256'])
        with os.fdopen(os.open(permit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
            import json
            json.dump(permit, stream)
        env = dict(os.environ, G4_PROBE_NONCE=nonce, CUDA_VISIBLE_DEVICES=gpu_uuid,
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                   HF_HUB_OFFLINE='1')
        with (run_dir / 'worker.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            report.update(status='RUNNING', worker_pid=child.pid)
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('hard deadline reached')
                if snapshot() - {child.pid}:
                    raise RuntimeError('foreign compute process detected')
                time.sleep(0.5)
        report['exit_code'] = child.returncode
        if child.returncode:
            raise RuntimeError('probe worker failed, see worker.log')
        if snapshot() - {child.pid}:
            raise RuntimeError('foreign compute process at final check')
        expected_status = ('QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED' if qkv_overlap else
                           'QKV_REPLAY_OUTPUT_VERIFIED' if qkv_replay else 'STANDALONE_PROBE_GPU_VERIFIED')
        if load(run_dir / 'result.json')['status'] != expected_status:
            raise RuntimeError('incomplete probe result')
        report['status'] = 'COMPLETED'
    except BaseException as error:
        report.update(status='STOPPED_OR_FAILED', error=str(error))
        if child and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=3)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        raise
    finally:
        if permit_path.exists():
            permit_path.unlink()
        report['end'] = dt.datetime.now().astimezone().isoformat()
        save(run_dir / 'supervision.json', report)


if __name__ == '__main__':
    main()
