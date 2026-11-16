"""A finite foreground plan. Each individual run has its own supervision/timeout."""
from pathlib import Path
import argparse,subprocess,json,datetime
P=Path.home()/'fp4-llm';O=P/'results/nsys_round2'
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline_d2','traces_d2','repeat_d2','nodes_d2','baseline_d1','traces_d1','repeat_d1','nodes_d1']);a=ap.parse_args()
batch=8 if a.stage.endswith('d2') else 1
if a.stage.startswith('baseline'):kinds=['baseline']
elif a.stage.startswith('traces'):kinds=['timeline','metrics']
elif a.stage.startswith('repeat'):kinds=['metrics']
else:kinds=['node']
jobs=[]
for kind in kinds:
    for mode in (['graph'] if kind=='node' or a.stage.startswith('repeat') else ['eager','graph']):
        for precision in ['bf16','nvfp4']:
            rep=3 if a.stage.startswith('repeat') else 2
            jobs.append(['python3',str(P/'scripts/run_one.py'),'--precision',precision,'--mode',mode,'--kind',kind,
                '--batch',str(batch),'--repeat',str(rep),'--steps','32' if kind=='node' else '256','--timeout','210'])
(O/'logs'/f'plan_{a.stage}.json').write_text(json.dumps(dict(created=datetime.datetime.now().astimezone().isoformat(),jobs=jobs),indent=2))
for i,cmd in enumerate(jobs):
    print('PLAN_JOB',a.stage,i+1,len(jobs),flush=True)
    r=subprocess.run(cmd,cwd=P)
    if r.returncode:raise SystemExit(r.returncode)
print('PLAN_COMPLETED',a.stage,flush=True)
