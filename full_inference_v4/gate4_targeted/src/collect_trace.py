"""Read one saved SQLite trace on the profiler host."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    host,rid=sys.argv[1:]
    dest=ROOT/'raw'/rid
    if dest.parent!=ROOT/'raw' or not (dest/'sqlite_manifest.json').exists():raise ValueError('missing trace')
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes',host,'python3','-',rid],input=(ROOT/'src/trace.py').read_bytes(),
                     capture_output=True,timeout=90)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
    data=json.loads(r.stdout)
    (dest/'validated_trace.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    from statistics import median
    for op in ('o','down'):
        rows=[r for r in data['rows'] if r['mode']==op+'_parallel']
        print(op,len(rows),sum(r['overlap_ns']>0 for r in rows),'overlap;',
              sum(r['fully_inside'] for r in rows),'fully inside;',median(r['fraction_b'] for r in rows))
if __name__=='__main__':main()
