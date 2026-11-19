"""Deploy one foreground run and retrieve its evidence from the existing supervisor."""
import argparse
import base64
import datetime as dt
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
import importlib.util
spec=importlib.util.spec_from_file_location('targeted_submit',ROOT.parent/'gate4_targeted/src/submit.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

LAUNCH="""
import base64,datetime,socket,sys,json,subprocess,tempfile
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
root=v4/'gate4_qk_ready'
if list((v4/'logs').glob('*_permit.json')):raise RuntimeError('active permit')
for n,body in sources.items():
    p=root/'src'/n
    p.parent.mkdir(parents=True,exist_ok=True)
    b=base64.b64decode(body)
    compile(b,n,'exec')
    if p.exists() and p.read_bytes()!=b:
        import hashlib
        old=p.read_bytes()
        archive=root/'archive'/(p.name+'.'+hashlib.sha256(old).hexdigest())
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():archive.write_bytes(old)
    p.write_bytes(b)
subprocess.run(['gpu-who'],check=True,timeout=10)
now=datetime.datetime.now().astimezone()
with tempfile.TemporaryDirectory(dir=root) as tmp:
    path=Path(tmp)/'window.json'
    path.write_text(json.dumps(dict(enabled=True,gate='4.4',hostname=socket.gethostname(),gpu_uuid='GPU-'+json.loads(
       (v4/'gate4_spatial/configs/hardware_from_gate3.json').read_bytes())['uuid'],
       allowed_workload=dict(experiment='E2',precision=prec,phase='decode_qk',condition='paired',profile=profile),
       not_before=now.isoformat(),expires_at=(now+datetime.timedelta(seconds=300)).isoformat(),
       authorization_reference='Run next real-consumer GPU experiment if shared device is free')))
    cmd=[sys.executable,str(root/'src/supervised.py'),'--config',str(v4/'configs'/('E2_'+prec+'.json')),
         '--run-id',rid,'--profile',profile,'--timeout','250','--window',str(path),'--execute']
    sys.exit(subprocess.run(cmd,timeout=310).returncode)
"""
def main():
    p=argparse.ArgumentParser()
    p.add_argument('host')
    p.add_argument('precision',choices=('nvfp4','bf16'))
    p.add_argument('profile',choices=('P0','P2'))
    p.add_argument('rep',choices=('r1','r2'))
    a=p.parse_args()
    rid=dt.datetime.now(dt.timezone.utc).strftime('qkready_%Y%m%d_%H%M%S_')+a.rep+'_'+a.precision+'_'+a.profile.lower()
    target=ROOT/'raw'/rid
    target.mkdir(parents=True,exist_ok=False)
    files=['worker.py','supervised.py','branches.py']
    sources={n:base64.b64encode((ROOT/'src'/n).read_bytes()).decode() for n in files}
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,'python3','-']
    prefix=f'rid={rid!r}\nprec={a.precision!r}\nprofile={a.profile!r}\n'
    print(rid,flush=True)
    run=subprocess.run(ssh,input=(prefix+f'sources={sources!r}\n'+LAUNCH).encode(),capture_output=True,timeout=350)
    fetch=subprocess.run(ssh,input=(prefix+prior.FETCH).encode(),capture_output=True,check=True,timeout=90)
    for name,val in json.loads(gzip.decompress(fetch.stdout)).items():
        if Path(name).name!=name:raise ValueError('invalid evidence name')
        (target/name).write_bytes(base64.b64decode(val))
    if not (target/'supervision.json').exists():
        (target/'precheck.txt').write_bytes(run.stderr)
    print(run.stdout.decode(errors='replace')[-350:],run.stderr.decode(errors='replace')[-550:])
    if run.returncode or not (target/'inference.json.gz').exists():
        log=target/(rid+'.log')
        if log.exists():print(log.read_text(errors='replace')[-2300:])
        raise SystemExit(1)
    record=json.load(gzip.open(target/'inference.json.gz'))
    ref=next((ROOT.parent/'gate4_targeted/raw').glob('*_r2_'+a.precision+'_p2/inference.json.gz'))
    base=json.load(gzip.open(ref))
    def hashes(trial):
        return sorted((r['prompt_hash'],r['output_hash'],r['prompt_length'],r['output_length']) for r in trial['request_summary'])
    if record.get('invocation_status')!='COMPLETED' or record['config_file_sha256']!=base['config_file_sha256']:
        raise ValueError('invalid complete E2 workload')
    for trial in record['runs']:
        if hashes(trial)!=hashes(base['runs'][0]) or len(trial['graph_replays'])!=256:
            raise ValueError('historical output or step count changed')
    print(len(record['runs']),'full E2 outputs validated')
if __name__=='__main__':
    main()
