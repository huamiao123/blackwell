"""Re-use the db46e14 acceptance checks, preserving the original analysis directory."""
import csv
import gzip
from pathlib import Path
import shutil
from statistics import mean
import sys
import tempfile
from analyze import ROOT,load,write

V4=ROOT.parent
E2=V4/'gate4_inference'
sys.path.insert(0,str(E2/'src'))
from summarize_dram_study import process

def main():
    raw=E2/'raw/dram_20261119'
    existing=E2/'analysis/dram_20261119'
    out=ROOT/'analysis/e2_dram'
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        scratch=Path(tmp)
        shutil.copytree(existing/'decode',scratch/'decode')
        process(raw,scratch)
        for name in ('statistics','paired_blocks','request_envelopes','target_latency_change','target_output_intervals',
                     'device_windows','probe_competition','run_inventory','failed_runs'):
            path=scratch/(name+'.csv')
            if path.exists():shutil.copyfile(path,out/path.name)
    if (out/'request_envelopes.csv').exists():
        with (out/'request_envelopes.csv').open() as f: measurements=list(csv.DictReader(f))
        comparison=[]
        import json
        for case in sorted((existing/'decode').glob('*.json')):
            prof=load(case)
            rid=prof['run_id']
            with gzip.open(raw/rid/(rid+'.json.gz'),'rt',encoding='utf8') as f:j=json.load(f)
            wall=j['runs'][0]['full_model_wall_ms']
            baseline=[float(r['model_wall_ms']) for r in measurements if r['precision']==prof['precision'] and r['condition']==prof['condition']]
            if not baseline:continue
            comparison.append(dict(precision=prof['precision'],condition=prof['condition'],p1_run=rid,p1_wall_ms=wall,
                                   p0_runs=len(baseline),p0_mean_wall_ms=mean(baseline),p1_minus_p0_pct=100*(wall/mean(baseline)-1)))
        if comparison:write(out/'profiler_comparison.csv',comparison)
if __name__=='__main__':
    main()
