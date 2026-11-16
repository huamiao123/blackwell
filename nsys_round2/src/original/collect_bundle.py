from pathlib import Path
import json,csv,statistics,re,hashlib,datetime
P=Path.home()/'fp4-llm';O=P/'results/nsys_round2';A=O/'analysis'
def savecsv(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summaries=[json.loads(f.read_text()) for f in sorted(A.glob('*_summary.json'))]
briefs=[];steps=[];telemetry=[]
for f in sorted((O/'raw').glob('*.json')):
    d=json.loads(f.read_text())
    if 'runs' not in d:continue
    brief={k:v for k,v in d.items() if k!='runs'};brief['file']=f.stem
    brief['runs']=[{k:v for k,v in r.items() if k not in ['scheduler','steps']} for r in d['runs']]
    briefs.append(brief)
    for r in d['runs']:
        for s in r['steps']:
            steps.append(dict(trace=f.stem,trial=r['trial'],precision=d['args']['precision'],batch=d['args']['batch'],mode=d['args']['mode'],
                protocol=d.get('protocol','sync_diagnostic_v1'),step=s['step'],latency_ms=s['latency_ms'],start_perf_counter_ns=s['start_ns'],end_perf_counter_ns=s['end_ns']))
    sf=O/'logs'/f'{f.stem}_supervision.json'
    if sf.exists():
        sv=json.loads(sf.read_text())
        for t in sv['telemetry']:
            text=t['status']
            def find(pattern):
                m=re.search(pattern,text);return float(m.group(1)) if m else None
            telemetry.append(dict(trace=f.stem,elapsed_s=t['elapsed'],
                used_memory_mib=find(r'FB Memory Usage\s*\n\s*Total[^\n]*\n\s*Reserved[^\n]*\n\s*Used\s*:\s*([\d.]+)'),
                sm_clock_mhz=find(r'\n    Clocks\s*\n\s*Graphics[^\n]*\n\s*SM\s*:\s*([\d.]+)'),
                temperature_c=find(r'GPU Current Temp\s*:\s*([\d.]+)'),power_w=find(r'Average Power Draw\s*:\s*([\d.]+)'),
                compute_pids=' '.join(map(str,t['compute_pids']))))
savecsv(A/'decode_step_stats.csv',steps);savecsv(A/'hardware_telemetry.csv',telemetry)
baselines={}
for d in briefs:
    if d.get('protocol')=='default_scheduler_v2' and '_baseline_' in d['file']:
        baselines.setdefault((d['args']['precision'],d['args']['batch'],d['args']['mode']),[]).extend(r['mean_step_ms'] for r in d['runs'])
overhead=[]
for d in briefs:
    if d.get('protocol')!='default_scheduler_v2' or '_baseline_' in d['file']:continue
    key=d['args']['precision'],d['args']['batch'],d['args']['mode']
    baseline=statistics.median(baselines[key]) if key in baselines else None
    measured=d['runs'][0]['mean_step_ms'];matched=d['args']['steps']==256
    overhead.append(dict(trace=d['file'],precision=key[0],batch=key[1],mode=key[2],steps=d['args']['steps'],
        baseline_median_step_ms=baseline,profile_step_ms=measured,
        vs_baseline_pct=(measured/baseline-1)*100 if baseline and matched else None,
        same_256_step_length=matched,graph_replays=d['runs'][0]['graph_replays']))
savecsv(A/'profiler_overhead.csv',overhead)
validations={}
for f in sorted(A.glob('*_validation.json')):
    if f.name=='timestamp_validation.json':continue
    d=json.loads(f.read_text());validations[f.stem]={k:v for k,v in d.items() if k in ['decode_window','gpu_inventory','cuda_devices','metric_sources','sampling','validation','activity_boundary_check','diagnostics','step_markers','kernels']}
(A/'timestamp_validation.json').write_text(json.dumps(validations,indent=2))
# Consolidate the requested primary CSVs, while keeping all per-trace data too.
primary=[s for s in summaries if s.get('protocol')=='default_scheduler_v2' and ('_metrics_' in s['trace'] or '_node_' in s['trace'])]
for suffix,dest,metric_filter in [('kernels','kernel_intervals.csv',False),('metrics','metric_samples.csv',True)]:
    writer=None
    with (A/dest).open('w',newline='') as out:
        for s in primary:
            f=A/f"{s['trace']}_{suffix}.csv"
            if not f.exists() or not f.stat().st_size:continue
            with f.open() as inp:
                reader=csv.DictReader(inp)
                if writer is None:writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader()
                for row in reader:
                    if metric_filter and not any(k in row['metric'] for k in ['SMs Active','SM Issue','Tensor Active','DRAM Read','DRAM Write','Compute Warps in Flight [Throughput']):continue
                    writer.writerow(row)
script_names=['decode_round2.py','decode_round2_sync_diagnostic.py','run_one.py','run_plan.py','validate_trace.py','analyze_round2.py','collect_bundle.py','audit_final.py','round2_make_figures.py','round2_make_report.py']
script_hashes={n:hashlib.sha256((P/'scripts'/n).read_bytes()).hexdigest() for n in script_names if (P/'scripts'/n).exists()}
bundle=dict(created=datetime.datetime.now().astimezone().isoformat(),summaries=summaries,raw_briefs=briefs,
    overhead=overhead,script_sha256=script_hashes,validations=validations)
(A/'report_data.json').write_text(json.dumps(bundle))
(O/'input_token_ids.json').write_text(json.dumps([1000+i%5000 for i in range(2048)]))
print('BUNDLE',len(summaries),'traces',len(briefs),'runs',len(steps),'step rows')
print('GPU_SCRIPT_HASHES',script_hashes)
