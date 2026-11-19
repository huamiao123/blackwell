"""Verify one measured NCU node and export each metric with its actual unit."""
import ast
import csv
import json
import math
from pathlib import Path
import re
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))
def read_run(path):
    precision,op,group,cache=load(path/'case.json')['case']
    info=load(path/'identity.json')
    live=info['live_identity']
    if (load(path/'supervision.json')['status']!='COMPLETED' or info['case']!=precision+'_'+op
        or len(info['trials'])!=3 or not all(t['exact_match'] for t in info['trials'])
        or len({t['output_sha256'] for t in info['trials']})!=1
        or load(path/'profiled_symbols.json')!=[live['kernel_name']]):
        raise ValueError(f'{path.name}: projection identity or output mismatch')
    with (path/'counters.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.reader(f))
    if len(rows)!=3 or any(len(r)!=len(rows[0]) for r in rows):
        raise ValueError('expected one NCU action and a units row')
    keys,units,values=rows
    data=dict(zip(keys,values))
    if (list(ast.literal_eval(data['Grid Size']))!=live['grid'] or
        list(ast.literal_eval(data['Block Size']))!=live['block']):
        raise ValueError('launch geometry differs from historical binding')
    passes=re.search(r'(\d+) pass(?:es)?\b',(path/'worker.log').read_text(errors='replace'))
    if not passes: raise ValueError('missing replay count')
    shared=dict(run_id=path.name,precision=precision,operator=op,cache_control=cache,
                counter_set=group,passes=int(passes[1]),grid_ctas=math.prod(live['grid']),
                homogeneous_blocks_per_sm=live['homogeneous_api_blocks_per_sm'])
    requested=load(ROOT/'configs/counters.json')[group]
    if any(k not in keys for k in requested): raise ValueError('missing requested counter')
    out=[]
    for k in requested:
        value=float(data[k].replace(',',''))
        if not math.isfinite(value): raise ValueError(k)
        status='OBSERVED'
        if 'sector_hit_rate.pct' in k and not 0 <= value <= 100:
            status='INVALID_HIT_PROBABILITY'
        out.append(dict(**shared,metric=k,unit=units[keys.index(k)],value=value,status=status))
    return out

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def main():
    out=ROOT/'analysis'
    out.mkdir(exist_ok=True)
    records=[]
    for path in sorted((ROOT/'raw').iterdir()):
        if (path/'report_manifest.json').exists():
            records.extend(read_run(path))
    if not records: raise RuntimeError('no verified runs')
    write(out/'measurements.csv',records)
    groups={}
    for row in records:
        if row['counter_set']!='pilot':
            key=tuple(row[k] for k in ('precision','operator','cache_control','counter_set','metric','unit'))
            groups.setdefault(key,[]).append(row)
    summary=[]
    for key,items in groups.items():
        base=dict(zip(('precision','operator','cache_control','counter_set','metric','unit'),key))
        vals=[r['value'] for r in items if r['status']=='OBSERVED']
        summary.append(dict(**base,processes=len(items),valid=len(vals),
                            median=median(vals) if vals else '',minimum=min(vals) if vals else '',
                            maximum=max(vals) if vals else ''))
    if summary: write(out/'summary.csv',summary)

if __name__=='__main__':
    main()
