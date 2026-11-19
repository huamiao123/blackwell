"""Export recorded supervision telemetry and exact worker source revisions without touching CUDA."""
import base64
import csv
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REMOTE=r"""
import base64,datetime,hashlib,json,re,sys
from pathlib import Path
v4=Path.home()/'fp4-llm/results/full_inference_v4'
root=v4/'gate4_qk_ready'
rows=[];versions={};old_blobs={}
for rid in sys.argv[1:]:
    if not re.fullmatch('qkready_[a-z0-9_]+',rid):raise ValueError('bad run id')
    record=json.loads((v4/'logs'/(rid+'_supervision.json')).read_bytes())
    row=dict(run=rid,status=record['status'],start=record['start'],end=record['end'])
    start=datetime.datetime.fromisoformat(row['start']).timestamp()
    version={}
    for name in ('worker.py','branches.py','supervised.py'):
        archive=sorted((p for p in (root/'archive').glob(name+'.*') if p.stat().st_mtime>start),key=lambda p:p.stat().st_mtime)
        b=(archive[0] if archive else root/'src'/name).read_bytes()
        version[name]=hashlib.sha256(b).hexdigest()
        if archive:old_blobs[name+':'+version[name]]=base64.b64encode(b).decode()
    versions[rid]=version
    patterns={'temp_c':r'^        GPU Current Temp\s*:\s*([0-9.]+) C','power_w':r'^        Instantaneous Power Draw\s*:\s*([0-9.]+) W','sm_mhz':r'^    Clocks\s*\n(?:        [^\n]+\n){0,2}?        SM\s*:\s*([0-9.]+) MHz'}
    samples=[s for s in record['telemetry'] if s['compute_pids']]
    row['active_samples']=len(samples)
    for key,pattern in patterns.items():
        vs=[float(m[1]) for s in samples if (m:=re.search(pattern,s['gpu'],re.M))]
        row[key+'_min']=min(vs) if vs else 'UNAVAILABLE'
        row[key+'_max']=max(vs) if vs else 'UNAVAILABLE'
    rows.append(row)
print(json.dumps(dict(telemetry=rows,source=versions,archived=old_blobs)))
"""

def main():
    ids=sorted(p.parent.name for p in (ROOT/'raw').glob('*/supervision.json'))
    v=json.loads(subprocess.check_output(['ssh','-T','-o','BatchMode=yes',sys.argv[1],'python3','-',*ids],input=REMOTE.encode(),timeout=50))
    out=ROOT/'analysis'
    with (out/'telemetry.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(v['telemetry'][0]),lineterminator='\n')
        w.writeheader();w.writerows(v['telemetry'])
    (out/'worker_revisions.json').write_text(json.dumps(v['source'],indent=2)+'\n',encoding='utf8')
    for key,b64 in v['archived'].items():
        name,digest=key.split(':')
        old=base64.b64decode(b64)
        if hashlib.sha256(old).hexdigest()!=digest:raise ValueError('archive hash mismatch')
        body=(ROOT/'src'/name).read_bytes()
        patch=difflib.unified_diff(body.decode().splitlines(True),old.decode().splitlines(True),'src/'+name,'src/'+name)
        (out/(name+'.'+digest[:10]+'.patch')).write_bytes(''.join(patch).encode())
if __name__=='__main__':
    main()
