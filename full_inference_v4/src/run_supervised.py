"""One bounded, supervised Gate-2 invocation. Default plan mode is CPU-only.

Existing ownership/timeout mechanism is retained and blocked prechecks are now
saved before launch. No polling for an idle window and no next-gate automation.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import time
from common import ROOT,load,save,file_hash
from collect_nsys import profiler_command,export_command
from run_full_inference import prepare


def validate_window(w,c,now,hostname):
    if not w.get('enabled'):raise ValueError('GPU_WINDOW_DISABLED')
    if w.get('gate')!=2 or w.get('reviewed_gates')!=[0,1]:raise ValueError('gate review missing')
    if c['experiment'] not in ['E1','E2'] or c['experiment']+'_'+c['precision'] not in w['allowed_configs']:
        raise ValueError('configuration outside reviewed scope')
    if w['hostname']!=hostname:raise ValueError('wrong host')
    begin=dt.datetime.fromisoformat(w['not_before']);finish=dt.datetime.fromisoformat(w['expires_at'])
    if begin.tzinfo is None or finish.tzinfo is None:raise ValueError('timezone required')
    start=begin.timestamp();end=finish.timestamp()
    if not start<=now<end:raise ValueError('outside authorized window')
    if not w.get('authorization_reference'):raise ValueError('missing explicit authorization reference')
    return end


def parse_gpu_state(text):
    pids=[]
    for block in re.split(r'\n\s+Process ID\s*:',text)[1:]:
        if re.search(r'Type\s*:\s*C',block):pids.append(int(block.splitlines()[0].strip()))
    recovery=re.search(r'GPU Recovery Action\s*:\s*(.+)',text)
    if recovery and recovery[1].strip()!='None':raise RuntimeError('GPU recovery needed')
    return pids


def controllers(ps):
    return any('wait_and_launch_live_controller' in line and 'python' in line for line in ps.splitlines())


def gpu_state():
    r=subprocess.run(['sudo','-n','/usr/bin/nvidia-smi','-q'],capture_output=True,text=True,timeout=10)
    if r.returncode:raise RuntimeError('GPU state unavailable: '+r.stderr)
    return r.stdout,parse_gpu_state(r.stdout)


def environment():
    project=Path.home()/'fp4-llm'
    sourced=subprocess.check_output(['bash','-c','source "$1"; env -0','v41-env',str(project/'env.sh')])
    env=os.environ.copy();env.update(dict(x.decode().split('=',1) for x in sourced.split(b'\0') if b'=' in x))
    selected=dict(PRJ=str(project),HOME=str(Path.home()),TMPDIR=str(project/'tmp'),TMP=str(project/'tmp'),TEMP=str(project/'tmp'),
                  VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',VLLM_NO_USAGE_STATS='1',TOKENIZERS_PARALLELISM='false',
                  TRITON_CACHE_DIR=str(project/'cache/triton'),VLLM_CACHE_ROOT=str(project/'cache/vllm'),
                  HF_HOME=str(project/'cache/hf'),XDG_CACHE_HOME=str(project/'cache'),MAX_JOBS='2',
                  FLASHINFER_WORKSPACE_BASE=str(project),FLASHINFER_AUTOTUNE_CACHE_DIR=str(project/'cache/flashinfer_autotune'),
                  TORCHINDUCTOR_CACHE_DIR=str(project/'cache/torchinductor'),TORCH_EXTENSIONS_DIR=str(project/'cache/torch_extensions'),
                  CUDA_CACHE_PATH=str(project/'cache/cuda'),PATH='/usr/local/cuda-13.3/bin:'+env['PATH'])
    env.update(selected)
    return env,selected


def main(profile_choices=('P0','P1','P2')):
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--profile',choices=profile_choices,default=profile_choices[0]);ap.add_argument('--timeout',type=int,default=420)
    ap.add_argument('--window',default=str(ROOT/'configs/gpu_window.json'));ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();c,ps=prepare(a.config)
    if not re.fullmatch('[a-z0-9_]+',a.run_id):raise ValueError('unsafe run ID')
    if not 1<=a.timeout<=900:raise ValueError('timeout')
    if not a.execute:
        print(json.dumps(dict(mode='CPU_PLAN_ONLY',gpu_queries=False,config=c['experiment'],precision=c['precision'],
                              profile=a.profile,requires='reviewed Gate 0/1 + explicit bounded GPU window'),indent=2));return
    # Validate window BEFORE any GPU query, torch import, subprocess workload or profiler.
    deadline=validate_window(load(a.window),c,time.time(),socket.gethostname())
    project=Path.home()/'fp4-llm'
    if not ROOT.resolve().is_relative_to(project.resolve()):raise RuntimeError('remote project root required')
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    recpath=logs/(a.run_id+'_supervision.json');dest=ROOT/'raw'/(a.run_id+'.json')
    if recpath.exists() or dest.exists():raise RuntimeError('refuse overwrite')
    record=dict(run_id=a.run_id,start=dt.datetime.now().astimezone().isoformat(),config_sha256=file_hash(a.config),
                args=vars(a),status='PRECHECK',command=None,telemetry=[],gpu_workload_started=False)
    save(recpath,record)
    permit_path=logs/(a.run_id+'_permit.json')
    child=None;known=set()
    try:
        q,pids=gpu_state();ps_text=subprocess.check_output(['ps','-eo','pid=,ppid=,uid=,comm=,args='],text=True)
        record['precheck_gpu']=q
        if pids:raise RuntimeError('GPU_BUSY_NO_RUN '+str(pids))
        if controllers(ps_text):raise RuntimeError('AUTO_CONTROLLER_PRESENT_NO_RUN')
        uuid=load(a.window)['gpu_uuid']
        if uuid not in q:raise RuntimeError('GPU UUID mismatch')
        env,selected=environment();nonce=secrets.token_hex(16)
        save(permit_path,dict(nonce=nonce,run_id=a.run_id,config_sha256=file_hash(a.config),expires_unix=min(deadline,time.time()+a.timeout),
                              supervisor_pid=os.getpid()))
        selected.update(V41_PERMIT_FILE=str(permit_path),V41_PERMIT_NONCE=nonce)
        env.update(selected)
        worker=[str(project/'vllm/.venv/bin/python'),str(ROOT/'src/run_full_inference.py'),'--config',str(Path(a.config).resolve()),
                '--run-id',a.run_id,'--profile',a.profile,'--output',str(dest),'--execute']
        cmd=profiler_command(worker,a.profile,ROOT/'raw'/a.run_id,selected)
        # Never publish nonce/ephemeral permit contents in runtime command logs.
        record['command']=[x if not x.startswith('V41_PERMIT_NONCE=') else 'V41_PERMIT_NONCE=<ephemeral>' for x in cmd]
        record['environment']={k:v for k,v in selected.items() if k!='V41_PERMIT_NONCE'}
        record['status']='RUNNING';save(recpath,record,overwrite=True)
        with (logs/(a.run_id+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            known.add(child.pid);record['gpu_workload_started']=True;t0=time.monotonic()
            while child.poll() is None:
                if time.monotonic()-t0>a.timeout or time.time()>=deadline:raise RuntimeError('HARD_TIMEOUT_OR_WINDOW_END')
                q,pids=gpu_state()
                ps_text=subprocess.check_output(['ps','-eo','pid=,ppid=,uid=,comm=,args='],text=True)
                rows=[l.split(None,4) for l in ps_text.splitlines()]
                for _ in range(12):
                    for row in rows:
                        if len(row)>=3 and int(row[1]) in known:known.add(int(row[0]))
                record['telemetry'].append(dict(monotonic_ns=time.perf_counter_ns(),gpu=q,compute_pids=pids))
                if set(pids)-known or controllers(ps_text):raise RuntimeError('FOREIGN_GPU_TASK_OR_CONTROLLER_APPEARED')
                time.sleep(2)
        record['returncode']=child.returncode
        if child.returncode:raise RuntimeError('WORKLOAD_FAILED')
        record['status']='COMPLETED'
        if a.profile in ('P1','P2'):
            rep=ROOT/'raw'/(a.run_id+'.nsys-rep');db=ROOT/'sqlite'/(a.run_id+'.sqlite')
            db.parent.mkdir(exist_ok=True)
            if not rep.is_file() or rep.stat().st_size==0:raise RuntimeError('EMPTY_TRACE')
            r=subprocess.run(export_command(rep,db),capture_output=True,text=True,timeout=90,env=env)
            (logs/(a.run_id+'_export.log')).write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError('EXPORT_FAILED')
    except BaseException as e:
        record.update(status='STOPPED_OR_FAILED',error=str(e))
        if child and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        raise
    finally:
        record.update(end=dt.datetime.now().astimezone().isoformat(),known_child_pids=sorted(known))
        save(recpath,record,overwrite=True)
        if permit_path.exists():permit_path.unlink()


if __name__=='__main__':main()
