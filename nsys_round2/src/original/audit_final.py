from pathlib import Path
import json,sqlite3,hashlib,collections
P=Path.home()/'fp4-llm';O=P/'results/nsys_round2';A=O/'analysis'
issues=[];baseline_configs=set();prompt_hashes=set();checks=[]
for f in (O/'raw').glob('*.json'):
    d=json.loads(f.read_text())
    if d.get('protocol')!='default_scheduler_v2':continue
    a=d['args'];prompt_hashes.add(d['prompt_sha256'])
    if not d['async_scheduling']:issues.append([f.name,'formal_async_not_default_true'])
    for r in d['runs']:
        if r['actual_batches']!=[a['batch']] or r['output_counts']!=[a['steps']+1]*a['batch']:issues.append([f.name,'batch_or_output_mismatch'])
        if len(r['steps'])!=a['steps']:issues.append([f.name,'step_count'])
        if r['graph_replays']!=(a['steps'] if a['mode']=='graph' else 0):issues.append([f.name,'graph_replays'])
    if '_baseline_' in f.stem:
        baseline_configs.add((a['batch'],a['precision'],a['mode']))
        if len(d['runs'])<5:issues.append([f.name,'too_few_trials'])
expected={(b,p,m) for b in [1,8] for p in ['bf16','nvfp4'] for m in ['eager','graph']}
if baseline_configs!=expected:issues.append(['baseline','missing_configs',list(expected-baseline_configs)])
if len(prompt_hashes)!=1:issues.append(['inputs','different_prompt_hashes',list(prompt_hashes)])
for f in A.glob('*_summary.json'):
    d=json.loads(f.read_text())
    if d.get('protocol')!='default_scheduler_v2':continue
    if d['kernel_coverage_reliable']:
        db=O/'sqlite'/f"{d['trace']}.sqlite";c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
        lo,hi=d['analysis_start_ns'],d['analysis_end_ns']
        edges=[]
        for start,end in c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where start<? and end>?',(hi,lo)):
            edges.extend([(max(start,lo),1),(min(end,hi),-1)])
        edges.sort();level=0;last=lo;total=0
        for t,delta in edges:
            if level:total+=t-last
            level+=delta;last=t
        c.close()
        diff=abs(total/1e6-d['kernel_union_ms'])
        if diff>1e-9:issues.append([f.name,'independent_union_mismatch',diff])
        checks.append(dict(trace=d['trace'],union_sweep_match=diff<=1e-9,coverage=d['kernel_coverage']))
    elif d['args']['mode']=='graph' and d['kernel_union_ms'] is not None:
        issues.append([f.name,'graph_envelope_used_as_kernel_union'])
    for m in d['metrics']:
        if d['kernel_coverage_reliable'] and m['active_samples']+m['gap_samples']!=m['valid_samples']:
            issues.append([f.name,'metric_classification_counts',m['name']])
files=[]
for f in sorted(O.rglob('*')):
    if f.is_file():files.append(dict(path=str(f.relative_to(O)),bytes=f.stat().st_size))
result=dict(issues=issues,baseline_configs=[list(x) for x in sorted(baseline_configs)],formal_prompt_hashes=sorted(prompt_hashes),independent_sweep_checks=checks,artifact_count=len(files))
(A/'quality_audit.json').write_text(json.dumps(result,indent=2))
(A/'artifact_manifest.json').write_text(json.dumps(files,indent=2))
print(json.dumps(result,indent=2))
if issues:raise SystemExit(2)
