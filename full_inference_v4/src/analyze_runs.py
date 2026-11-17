"""Collect only measured full-inference JSON; profiling comparisons require hashes."""
import argparse
from pathlib import Path
from common import ROOT,load,csv_write,digest


def compare_overhead(baseline,profile):
    for key in ['config_file_sha256','prompt_manifest_hash','code_sha256']:
        if baseline[key]!=profile[key]:raise ValueError('unmatched '+key)
    b=sum(r['full_model_wall_ms'] for r in baseline['runs'])/len(baseline['runs'])
    return dict(profile=profile['profile'],baseline_mean_ms=b,
                profiled_mean_ms=sum(r['full_model_wall_ms'] for r in profile['runs'])/len(profile['runs']),
                overhead_ratio=(sum(r['full_model_wall_ms'] for r in profile['runs'])/len(profile['runs']))/b-1)


def analyze(root=ROOT):
    latency=[];iterations=[];paired=[]
    for f in sorted((root/'raw').glob('*.json')):
        d=load(f)
        if d.get('schema')!='v4.1' or 'runs' not in d or d.get('invocation_status')!='COMPLETED':continue
        for run in d['runs']:
            for r in run['request_summary']:
                latency.append(dict(source=f.name,precision=d['config']['precision'],experiment=d['config']['experiment'],
                                    profile=d['profile'],trial=run['trial'],**r))
            for it in run['scheduler_iterations']:
                iterations.append(dict(source=f.name,trial=run['trial'],**{k:v for k,v in it.items() if k!='requests'},
                                       request_assignment_json=str(it['requests'])))
    if latency:csv_write(root/'analysis/e2e_latency.csv',latency)
    if iterations:csv_write(root/'analysis/scheduler_iterations.csv',iterations)
    return len(latency)


if __name__=='__main__':print('Measured request rows:',analyze())
