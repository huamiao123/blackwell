"""Submit one supervised diagnostic via SSH. No idle polling or retries."""
import argparse
import base64
import datetime as dt
import gzip
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT.parent
REMOTE = """
import datetime, json, socket, subprocess, sys, tempfile
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
root=v4/'gate4_microarch'
if list((v4/'logs').glob('*_permit.json')):
    sys.exit('existing supervisor permit')
now=datetime.datetime.now().astimezone()
uuid=json.loads((v4/'gate4_spatial/configs/hardware_from_gate3.json').read_text())['uuid']
with tempfile.TemporaryDirectory(prefix='micro-window-',dir=root) as folder:
    w=Path(folder)/'window.json'
    w.write_text(json.dumps(dict(enabled=True,gate='microarch',hostname=socket.gethostname(),
        gpu_uuid='GPU-'+uuid.removeprefix('GPU-'),not_before=now.isoformat(),
        expires_at=(now+datetime.timedelta(seconds=195)).isoformat(),allowed=case,
        authorization_reference='User requested continuing supervised projection microarchitecture measurements')))
    subprocess.run(['gpu-who'],timeout=10,check=True)
    precision,operator,counters,cache=case
    cmd=['python3',str(root/'src/run_profile.py'),'--operator',operator,'--counters',counters,'--cache',cache,
        '--run-id',rid,'--config',str(v4/'configs'/('E2_'+precision+'.json')),'--window',str(w),
        '--profile','P3','--timeout','165','--execute']
    sys.exit(subprocess.run(cmd,timeout=210).returncode)
"""
FETCH = """
import base64,gzip,hashlib,json,re,subprocess,sys
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
root=v4/'gate4_microarch'
out={}
def put(k,b):
    out[k]=base64.b64encode(b).decode()
for key,path in [('worker.log',v4/'logs'/(rid+'.log')),('identity.json',root/'raw'/rid/'result.json')]:
    if path.is_file():
        put(key,re.sub(rb'V41_PERMIT_NONCE=[a-fA-F0-9]+',b'V41_PERMIT_NONCE=<ephemeral>',path.read_bytes()))
record=v4/'logs'/(rid+'_supervision.json')
if record.exists():
    r=json.loads(record.read_text())
    # Raw device telemetry stays on the host. Publish timestamps, status and our command.
    put('supervision.json',json.dumps({k:r.get(k) for k in ('run_id','start','end','args','status','error','returncode','command','gpu_workload_started')}).encode())
rep=root/'reports'/(rid+'.ncu-rep')
if rep.is_file() and rep.stat().st_size:
    sys.path.insert(0,'/opt/nvidia/nsight-compute/2026.2.1/extras/python')
    import ncu_report
    ctx=ncu_report.load_report(str(rep))
    symbols=[ctx.range_by_idx(i).action_by_idx(j).name(ncu_report.IAction.NameBase_MANGLED)
             for i in range(ctx.num_ranges()) for j in range(ctx.range_by_idx(i).num_actions())]
    put('profiled_symbols.json',json.dumps(symbols).encode())
    if case[2]=='pc':
        action=ctx.range_by_idx(0).action_by_idx(0)
        evidence={}
        for name in ['smsp__pcsamp_sample_count','smsp__pcsamp_warps_issue_stalled_sleeping']:
            metric=action.metric_by_name(name)
            if not metric: continue
            record=dict(total=metric.as_uint64(),instances=[])
            if metric.has_correlation_ids():
                pcs=metric.correlation_ids()
                for i in range(metric.num_instances()):
                    addr=pcs.as_uint64(i)
                    count=metric.as_uint64(i)
                    if count:
                        record['instances'].append(dict(pc=hex(addr),samples=count,sass=action.sass_by_pc(addr),
                            previous_sass=action.sass_by_pc(addr-16)))
            evidence[name]=record
        put('pc_samples.json',json.dumps(evidence).encode())
    cmd=['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(rep),'--csv','--page','raw','--print-units','base','--kernel-name-base','mangled']
    res=subprocess.run(cmd,capture_output=True,timeout=60)
    if res.returncode: raise RuntimeError(res.stderr.decode(errors='replace'))
    put('counters.csv',res.stdout)
    put('report_manifest.json',json.dumps(dict(bytes=rep.stat().st_size,sha256=hashlib.sha256(rep.read_bytes()).hexdigest(),remote_path=str(rep))).encode())
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode()))
"""

def main():
    p=argparse.ArgumentParser()
    p.add_argument('host')
    p.add_argument('precision',choices=('nvfp4','bf16'))
    p.add_argument('operator',choices=('qkv_proj','o_proj','gate_up_proj','down_proj'))
    p.add_argument('counters',choices=('pilot','resources','stalls','cache_read','pc'))
    p.add_argument('--cache',choices=('none','all'),default='none')
    a=p.parse_args()
    case=[a.precision,a.operator,a.counters,a.cache]
    rid='micro_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+a.precision+'_'+a.operator
    dest=ROOT/'raw'/rid
    dest.mkdir(parents=True,exist_ok=False)
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host,'python3','-']
    prefix=f'rid={rid!r}\ncase={case!r}\n'
    print(rid,case,flush=True)
    (dest/'case.json').write_text(json.dumps(dict(case=case,run_id=rid)))
    run=subprocess.run(ssh,input=(prefix+REMOTE).encode(),timeout=240,capture_output=True)
    print('remote',run.returncode,run.stderr.decode(errors='replace')[-800:],flush=True)
    payload=subprocess.run(ssh,input=(prefix+FETCH).encode(),timeout=85,capture_output=True,check=True)
    for name,data in json.loads(gzip.decompress(payload.stdout)).items():
        if Path(name).name!=name: raise ValueError('unexpected artifact path')
        (dest/name).write_bytes(base64.b64decode(data))
    if (dest/'worker.log').exists():
        print((dest/'worker.log').read_text(errors='replace')[-1300:])
    if not (dest/'supervision.json').exists():
        (dest/'submit_error.txt').write_bytes(run.stderr)
    if run.returncode or not {'identity.json','counters.csv','report_manifest.json'} <= {f.name for f in dest.iterdir()}:
        raise SystemExit(1)
    from analyze import read_run
    read_run(dest)

if __name__=='__main__':
    main()
