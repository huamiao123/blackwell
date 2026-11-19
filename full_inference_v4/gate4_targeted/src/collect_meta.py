"""Read saved run telemetry and the exact source versions from the host. No GPU calls."""
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
root=v4/'gate4_targeted'
sources=['protocol.py','target_graph.py','target_worker.py','supervised.py']
blobs={}
rows=[]
for rid in sys.argv[1:]:
    if not re.fullmatch('nodetest_[a-z0-9_]+',rid):raise ValueError(rid)
    record=json.loads((v4/'logs'/(rid+'_supervision.json')).read_bytes())
    t=datetime.datetime.fromisoformat(record['start']).timestamp()
    versions={}
    for name in sources:
        old=sorted((p for p in (root/'archive').glob(name+'.*') if p.stat().st_mtime>t),key=lambda p:p.stat().st_mtime)
        body=(old[0] if old else root/'src'/name).read_bytes()
        digest=hashlib.sha256(body).hexdigest()
        versions[name]=digest
        blobs[name+':'+digest]=base64.b64encode(body).decode()
    samples=[]
    for ent in record.get('telemetry',[]):
        if not ent['compute_pids']:continue
        q=ent['gpu']
        matches=[re.search(pattern,q,re.M) for pattern in
            [r'^        GPU Current Temp\s*:\s*([\d.]+) C',r'^        Instantaneous Power Draw\s*:\s*([\d.]+) W',
             r'^    Clocks\s*\n(?:        [^\n]+\n){0,2}?        SM\s*:\s*([\d.]+) MHz']]
        samples.append([float(m[1]) if m else None for m in matches])
    rows.append(dict(run=rid,status=record['status'],start=record['start'],end=record['end'],source=versions,
                     monitored_samples=samples))
print(json.dumps(dict(rows=rows,blobs=blobs)))
"""
def main():
    host=sys.argv[1]
    ids=sorted(p.parent.name for p in (ROOT/'raw').glob('*/supervision.json'))
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes',host,'python3','-',*ids],input=REMOTE.encode(),capture_output=True,check=True,timeout=60)
    meta=json.loads(r.stdout)
    out=ROOT/'analysis'
    out.mkdir(exist_ok=True)
    for key,val in meta['blobs'].items():
        name,digest=key.split(':')
        body=base64.b64decode(val)
        if hashlib.sha256(body).hexdigest()!=digest:raise ValueError('source hash mismatch')
        current=(ROOT/'src'/name).read_bytes()
        if body!=current:
            archive=ROOT/'source_history';archive.mkdir(exist_ok=True)
            diff=difflib.unified_diff(current.decode().splitlines(keepends=True),body.decode().splitlines(keepends=True),
                                      fromfile='src/'+name,tofile='src/'+name)
            (archive/(name+'.'+digest[:12]+'.patch')).write_bytes(''.join(diff).encode())
    (out/'source_versions.json').write_text(json.dumps({r['run']:r['source'] for r in meta['rows']},indent=2)+'\n',encoding='utf8')
    rows=[]
    for r in meta['rows']:
        rec={k:r[k] for k in ('run','status','start','end')}
        rec['samples_with_compute_process']=len(r['monitored_samples'])
        for i,k in enumerate(('gpu_temp_c','power_w','sm_clock_mhz')):
            vals=[s[i] for s in r['monitored_samples'] if s[i] is not None]
            rec[k+'_min']=min(vals) if vals else 'UNAVAILABLE'
            rec[k+'_max']=max(vals) if vals else 'UNAVAILABLE'
        rows.append(rec)
    with (out/'supervision.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
