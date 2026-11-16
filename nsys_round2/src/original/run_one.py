"""One bounded, foreground, supervised run; stops on another compute process."""
from pathlib import Path
import argparse,subprocess,os,json,time,signal,datetime,re,shlex

P=Path.home()/'fp4-llm'; O=P/'results/nsys_round2'
def gpu_state():
    r=subprocess.run(['sudo','-n','/usr/bin/nvidia-smi','-q'],capture_output=True,text=True,timeout=10)
    if r.returncode:raise RuntimeError(r.stderr)
    processes=[]
    for b in re.split(r'\n\s+Process ID\s*:',r.stdout)[1:]:
        if re.search(r'Type\s*:\s*C',b):processes.append(int(b.splitlines()[0].strip()))
    if 'GPU Recovery Action' in r.stdout and not re.search(r'GPU Recovery Action\s*:\s*None',r.stdout):
        raise RuntimeError('GPU recovery action is not None')
    return r.stdout,processes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--precision',required=True,choices=['bf16','nvfp4'])
    ap.add_argument('--mode',required=True,choices=['eager','graph'])
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--kind',choices=['baseline','timeline','metrics','node','pilot'],required=True)
    ap.add_argument('--repeat',type=int,default=1)
    ap.add_argument('--steps',type=int,default=256)
    ap.add_argument('--frequency',type=int,default=10000)
    ap.add_argument('--timeout',type=int,default=240)
    a=ap.parse_args()
    os.chdir(P)
    env=os.environ.copy()
    sourced=subprocess.check_output(['bash','-c','source "$1"; env -0','round2-env',str(P/'env.sh')])
    env.update(dict(x.decode().split('=',1) for x in sourced.split(b'\0') if b'=' in x))
    env.update(PRJ=str(P),HOME=str(Path.home()),TMPDIR=str(P/'tmp'),TMP=str(P/'tmp'),TEMP=str(P/'tmp'),
        TRITON_CACHE_DIR=str(P/'cache/triton'),VLLM_CACHE_ROOT=str(P/'cache/vllm'),
        HF_HOME=str(P/'cache/hf'),HF_HUB_OFFLINE='1',VLLM_NO_USAGE_STATS='1',TOKENIZERS_PARALLELISM='false',
        VLLM_ENABLE_V1_MULTIPROCESSING='0',MAX_JOBS='2',XDG_CACHE_HOME=str(P/'cache'),
        FLASHINFER_WORKSPACE_BASE=str(P),FLASHINFER_AUTOTUNE_CACHE_DIR=str(P/'cache/flashinfer_autotune'),
        TORCHINDUCTOR_CACHE_DIR=str(P/'cache/torchinductor'),TORCH_EXTENSIONS_DIR=str(P/'cache/torch_extensions'),
        CUDA_CACHE_PATH=str(P/'cache/cuda'))
    env['PATH']='/usr/local/cuda-13.3/bin:'+env['PATH']
    q,pids=gpu_state()
    who=subprocess.run(['/usr/local/bin/gpu-who'],capture_output=True,text=True,timeout=10)
    if pids:raise RuntimeError('GPU_BUSY_NO_RUN '+str(pids))
    ps=subprocess.run(['ps','-eo','pid=,ppid=,uid=,comm=,args='],capture_output=True,text=True,check=True)
    if any('wait_and_launch_live_controller' in l and 'python' in l for l in ps.stdout.splitlines()):
        raise RuntimeError('AUTO_LAUNCH_CONTROLLER_PRESENT_NO_RUN')
    stem=f'{a.precision}_b{a.batch}_kv2048_{a.mode}_{a.kind}_{a.frequency}hz_r{a.repeat:02d}'
    output=O/'raw'/f'{stem}.json'
    assert not output.exists(),output
    command=[str(P/'vllm/.venv/bin/python'),str(P/'scripts/decode_round2.py'),'--precision',a.precision,
        '--mode',a.mode,'--batch',str(a.batch),'--steps',str(a.steps),'--trials','5' if a.kind=='baseline' else '1','--output',str(output)]
    if a.kind!='baseline':
        command+=['--profile']
        flags=['--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace='+('node' if a.kind=='node' else 'graph'),
            '--capture-range=cudaProfilerApi','--capture-range-end=stop-shutdown','--kill=none',
            '--output='+str(O/'raw'/stem)]
        if a.kind in ['metrics','pilot','node']:
            flags+=['--gpu-metrics-devices=0','--gpu-metrics-set=gb20x','--gpu-metrics-frequency='+str(a.frequency)]
        selected={k:v for k,v in env.items() if k in ['HOME','PRJ','TMPDIR','TMP','TEMP','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','HF_HOME','HF_HUB_OFFLINE','VLLM_NO_USAGE_STATS','TOKENIZERS_PARALLELISM','VLLM_ENABLE_V1_MULTIPROCESSING','MAX_JOBS','PATH','XDG_CACHE_HOME','FLASHINFER_WORKSPACE_BASE','FLASHINFER_AUTOTUNE_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH']}
        command=['sudo','-n','--preserve-env=TMPDIR','/usr/local/sbin/gpu-profile','nsys','profile',*flags,'env',*[f'{k}={v}' for k,v in selected.items()],*command]
    record=dict(start=datetime.datetime.now().astimezone().isoformat(),args=vars(a),command=command,
        environment={k:v for k,v in env.items() if k in ['PRJ','HOME','TMPDIR','TMP','TEMP','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','HF_HOME','HF_HUB_OFFLINE','VLLM_NO_USAGE_STATS','TOKENIZERS_PARALLELISM','VLLM_ENABLE_V1_MULTIPROCESSING','MAX_JOBS','PATH','XDG_CACHE_HOME']},gpu_who=who.stdout)
    (O/'logs'/f'{stem}_precheck.txt').write_text(q+'\n'+who.stdout)
    with (O/'commands.txt').open('a') as f:f.write(json.dumps(record)+'\n')
    print('START',stem,flush=True)
    log=O/'logs'/f'{stem}.log'
    telemetry=[];status='running'
    with log.open('w') as f:
        child=subprocess.Popen(command,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic()
        known_descendants={child.pid}
        try:
            while child.poll() is None:
                if time.monotonic()-start>a.timeout:raise RuntimeError('HARD_TIMEOUT')
                text,pids=gpu_state()
                tree=subprocess.run(['ps','-eo','pid=,ppid=,uid='],capture_output=True,text=True,check=True)
                rows=[tuple(map(int,l.split())) for l in tree.stdout.splitlines() if len(l.split())==3]
                descendants=set(known_descendants)
                for _ in range(12):
                    for pid,ppid,uid in rows:
                        if ppid in descendants:descendants.add(pid)
                known_descendants.update(descendants)
                other=[pid for pid in pids if pid not in known_descendants]
                telemetry.append(dict(elapsed=time.monotonic()-start,status=text,compute_pids=pids))
                if other:raise RuntimeError('OTHER_COMPUTE_TASK_APPEARED '+str(other))
                time.sleep(2)
            status='completed' if child.returncode==0 else 'failed'
        except BaseException as e:
            status=str(e)
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    record.update(status=status,returncode=child.returncode,end=datetime.datetime.now().astimezone().isoformat())
    (O/'logs'/f'{stem}_supervision.json').write_text(json.dumps(dict(record=record,telemetry=telemetry),indent=2))
    text,pids=gpu_state()
    for _ in range(10):
        if not pids:break
        time.sleep(1)
        text,pids=gpu_state()
    (O/'logs'/f'{stem}_postcheck.txt').write_text(text)
    print('END',status,'RC',child.returncode,'COMPUTE_PIDS',pids,flush=True)
    loglines=log.read_text().splitlines()
    concise=[l for l in loglines if l.startswith('ROUND2_') or 'Traceback' in l or 'Error:' in l or 'ERROR' in l]
    print('\n'.join(concise[-15:]) or '\n'.join(loglines[-5:]))
    if child.returncode or status!='completed':raise SystemExit(4)
    if a.kind!='baseline':
        rep=O/'raw'/f'{stem}.nsys-rep'
        if not rep.is_file():raise RuntimeError('EMPTY_TRACE_MISSING_REPORT')
        dest=O/'sqlite'/f'{stem}.sqlite'
        cmd=['nsys','export','--type=sqlite','--output='+str(dest),str(rep)]
        with (O/'commands.txt').open('a') as f:f.write(json.dumps(dict(command=cmd))+'\n')
        r=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=90)
        (O/'logs'/f'{stem}_export.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('EXPORT_FAILED '+r.stdout+r.stderr)
        print('EXPORTED',dest,flush=True)
        from validate_trace import validate
        validation=validate(dest)
        (O/'analysis'/f'{stem}_validation.json').write_text(json.dumps(validation,indent=2))
        assert validation['step_markers']==a.steps,('STEP_MARKER_COUNT',validation['step_markers'])
        assert validation['validation']['gpu_identity_match']
        graph_calls=sum(r['count'] for r in validation['runtime_api_counts'] if 'cudaGraphLaunch' in r['name'])
        assert graph_calls==(a.steps if a.mode=='graph' else 0),('ACTUAL_GRAPH_MODE',graph_calls)
        if a.kind in ['metrics','pilot','node']:
            assert validation.get('sampling',{}).get('inside_window',0)>0,'EMPTY_METRICS'
            assert validation['sampling']['intervals_gt_1_5x_median']==0,'METRICS_SAMPLE_GAPS'
        severe=[r for r in validation['diagnostics'] if r['severity']>=3 or re.search(r'overflow|missing data|data loss',r['text'],re.I)]
        assert not severe,('NSYS_DIAGNOSTIC_FAILURE',severe)
        print('TRACE_GATE_PASSED',stem,flush=True)

if __name__=='__main__':main()
