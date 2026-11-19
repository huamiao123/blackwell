"""Summarize reproducible, per-process stage competition and neighboring GPU activities."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, old_summary):
    evidence = [json.loads(path.read_bytes()) for path in sorted((root/'neighbors').glob('*.json'))]
    with old_summary.open(encoding='utf-8') as f:
        previous = [r for r in csv.DictReader(f) if r['phase_condition']=='lifecycle']
    old = {r[k] for r in previous for k in ('parallel_run','serial_run')}
    groups = {}
    shares, results = [], []
    for e in evidence:
        cohort = 'original' if e['run_id'] in old else 'independent_repeat'
        key = cohort, e['precision'], e['condition']
        if key in groups:
            raise ValueError('duplicate trace')
        groups[key] = e
    for (cohort,precision,condition), e in groups.items():
        if condition!='parallel':
            continue
        serial=groups[cohort,precision,'serial']
        for stage,expected in [('PREFILL',1),('DECODE',32)]:
            probes=[r for r in e['rows'] if r['phase']==stage]
            controls=[r for r in serial['rows'] if r['phase']==stage]
            if len(probes)!=expected or [r['iteration'] for r in probes]!=[r['iteration'] for r in controls]:
                raise ValueError('incomplete paired work')
            if any(r['overlap_ns'] for r in controls):
                raise ValueError('serial control overlaps')
            def durations(rows):
                return [(r['probe_end_ns']-r['probe_start_ns'])/1000 for r in rows]
            p,s=durations(probes),durations(controls)
            total=sum(r['overlap_ns'] for r in probes)
            fam=defaultdict(int)
            for r in probes:
                for h in r['hits']:
                    n=e['names'][str(h['name_id'])]
                    fam[n['short'],n['demangled'],tuple(h['grid']),tuple(h['block'])]+=h['overlap_ns']
            if sum(fam.values())!=total:
                raise ValueError('overlapping primary kernel families cannot be stacked')
            common=dict(cohort=cohort,precision=precision,stage=stage,parallel_run=e['run_id'],serial_run=serial['run_id'])
            for (name,full,grid,block),n in sorted(fam.items(),key=lambda pair:-pair[1]):
                shares.append(dict(**common,kernel_short=name,kernel_demangled=full,grid=str(grid),block=str(block),
                                   overlap_us=n/1000,share_of_verified_overlap=n/total))
            results.append(dict(**common,probe_calls=len(p),overlapping_calls=sum(r['overlap_ns']>0 for r in probes),
                                serial_overlapping_calls=0,work_tiles_each=256,parallel_median_us=median(p),
                                serial_median_us=median(s),conditional_throughput_ratio=sum(s)/sum(p),
                                subset_overlap_us=total/1000,
                                start_rel_to_primary_us=median((r['probe_start_ns']-r['primary_first_ns'])/1000 for r in probes),
                                launch_api_start_to_gpu_start_us=median((r['probe_start_ns']-r['api_start_ns'])/1000 for r in probes)))
    if len(results)!=8:
        raise ValueError('require original and independent repeat for both precisions and phases')
    write(root/'phase_comparison.csv',results)
    write(root/'overlapping_families.csv',shares)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('analysis',type=Path)
    parser.add_argument('original_summary',type=Path)
    args=parser.parse_args()
    summarize(args.analysis,args.original_summary)
