"""Retrieve the checked node timeline, leaving the large SQLite on the lab server."""
import gzip
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
host,rid=sys.argv[1:]
p=root/'raw'/rid
if p.parent!=root/'raw' or not (p/'sqlite_manifest.json').exists():
    raise ValueError('missing local manifest')
r=subprocess.run(['ssh','-T','-o','BatchMode=yes',host,'python3','-',rid],input=(root/'src/extract.py').read_bytes(),
                 capture_output=True,timeout=80)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
trace=json.loads(gzip.decompress(r.stdout))
if trace['run']!=rid:raise ValueError('run mismatch')
(p/'intervals.json.gz').write_bytes(r.stdout)
for mode in ('original','sham','fork'):
    layers=[g for s in trace['replays'] if s['mode']==mode for g in s['layers']]
    print(mode,len(layers),sum(min(r[3],r[5])>max(r[2],r[4]) for r in layers),'overlap')
