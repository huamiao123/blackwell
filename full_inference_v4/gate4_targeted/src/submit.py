"""Deploy the worker, run one bounded foreground job and retrieve its evidence."""
import base64
import datetime
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REMOTE="""
import base64,datetime,hashlib,json,socket,subprocess,tempfile
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
root=v4/'gate4_targeted'
if list((v4/'logs').glob('*_permit.json')):raise RuntimeError('existing supervisor permit')
for name,data in sources.items():
    p=root/'src'/name
    p.parent.mkdir(parents=True,exist_ok=True)
    body=base64.b64decode(data)
    compile(body,name,'exec')
    if p.exists() and p.read_bytes()!=body:
        old=p.read_bytes()
        archive=root/'archive'/(p.name+'.'+hashlib.sha256(old).hexdigest())
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():archive.write_bytes(old)
    p.write_bytes(body)
subprocess.run(['gpu-who'],check=True,timeout=10)
now=datetime.datetime.now().astimezone()
with tempfile.TemporaryDirectory(dir=root,prefix='window-') as tmp:
    w=Path(tmp)/'window.json'
    w.write_text(json.dumps(dict(enabled=True,gate='4.4',hostname=socket.gethostname(),
        gpu_uuid='GPU-'+json.loads((v4/'gate4_spatial/configs/hardware_from_gate3.json').read_text())['uuid'],
        allowed_workload=dict(experiment='E2',precision=prec,phase='decode_nodes',condition='paired',profile=profile),
        not_before=now.isoformat(),expires_at=(now+datetime.timedelta(seconds=400)).isoformat(),
        authorization_reference='Continue the supervised E2 Down and O placement comparison')))
    args=['python3',str(root/'src/supervised.py'),'--config',str(v4/'configs'/('E2_'+prec+'.json')),
          '--profile',profile,'--run-id',rid,'--window',str(w),'--timeout','345','--execute']
    raise SystemExit(subprocess.run(args,timeout=475).returncode)
"""
FETCH="""
import base64,gzip,hashlib,json,re,sys
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
out={}
def add(n,b):
    out[n]=base64.b64encode(b).decode()
for suffix in ('.json','_failure.json'):
    p=v4/'raw'/(rid+suffix)
    if p.exists():add('inference'+suffix+'.gz',gzip.compress(p.read_bytes(),mtime=0))
for suffix in ('.log','_export.log'):
    p=v4/'logs'/(rid+suffix)
    if p.exists():add(p.name,re.sub(rb'V41_PERMIT_NONCE=[a-fA-F0-9]+',b'V41_PERMIT_NONCE=<ephemeral>',p.read_bytes()))
p=v4/'logs'/(rid+'_supervision.json')
if p.exists():
    j=json.loads(p.read_bytes())
    add('supervision.json',json.dumps({k:j.get(k) for k in ('run_id','start','end','status','error','command','returncode','gpu_workload_started','config_sha256')}).encode())
for folder,suffix in [('raw','.nsys-rep'),('sqlite','.sqlite')]:
    p=v4/folder/(rid+suffix)
    if p.exists():
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        add(suffix[1:]+'_manifest.json',json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=h)).encode())
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode()))
"""
def main():
    host,prec,profile,rep=sys.argv[1:]
    if prec not in ('bf16','nvfp4') or profile not in ('P0','P2') or rep not in ('r1','r2','r3'):raise ValueError('unexpected case')
    rid='nodetest_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_')+rep+'_'+prec+'_'+profile.lower()
    files=['protocol.py','target_graph.py','target_worker.py','supervised.py']
    bodies={n:(ROOT/'src'/n).read_bytes() for n in files}
    for n,b in bodies.items():compile(b,n,'exec')
    sources={n:base64.b64encode(b).decode() for n,b in bodies.items()}
    prefix=f'rid={rid!r}\nprec={prec!r}\nprofile={profile!r}\n'
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3','-']
    dest=ROOT/'raw'/rid
    dest.mkdir(parents=True,exist_ok=False)
    print(rid,flush=True)
    proc=subprocess.run(ssh,input=(prefix+f'sources={sources!r}\n'+REMOTE).encode(),timeout=495,capture_output=True)
    print(proc.stdout.decode(errors='replace')[-500:],proc.stderr.decode(errors='replace')[-500:],flush=True)
    got=subprocess.run(ssh,input=(prefix+FETCH).encode(),timeout=125,capture_output=True,check=True)
    for n,data in json.loads(gzip.decompress(got.stdout)).items():
        if Path(n).name!=n:raise ValueError('invalid artifact name')
        (dest/n).write_bytes(base64.b64decode(data))
    if not (dest/'supervision.json').exists():(dest/'precheck_error.txt').write_bytes(proc.stderr)
    log=dest/(rid+'.log')
    if log.exists():print(log.read_text(errors='replace')[-1800:])
    if proc.returncode or not (dest/'inference.json.gz').exists():raise SystemExit(1)
if __name__=='__main__':main()
