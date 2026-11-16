"""CPU-only analysis of actual SQLite schema; never fabricates timestamp offsets."""
from pathlib import Path
import sqlite3,json,csv,math,statistics,bisect,collections,argparse
from validate_trace import validate

P=Path.home()/'fp4-llm';O=P/'results/nsys_round2'
def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out
def percentile(v,q):
    if not v:return None
    s=sorted(v);x=(len(s)-1)*q;l=int(x);h=min(l+1,len(s)-1)
    return s[l]+(s[h]-s[l])*(x-l)
def csv_write(path,rows,fields=None):
    if not rows:
        path.write_text('');return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
_interval_cache={}
def overlap_length(intervals,a,b):
    key=id(intervals)
    if key not in _interval_cache:
        starts=[x for x,y in intervals];ends=[y for x,y in intervals];prefix=[0]
        for x,y in intervals:prefix.append(prefix[-1]+y-x)
        _interval_cache[key]=(intervals,starts,ends,prefix)
    _,starts,ends,prefix=_interval_cache[key]
    def measure(t):
        i=bisect.bisect_right(starts,t)-1
        return 0 if i<0 else prefix[i]+max(0,min(t,ends[i])-starts[i])
    return measure(b)-measure(a)
def analyze_one(path):
    v=validate(path)
    c=sqlite3.connect(f'file:{path}?mode=ro',uri=True);c.row_factory=sqlite3.Row
    raw=json.loads((O/'raw'/f'{path.stem}.json').read_text());args=raw['args']
    capture_start,capture_end=v['decode_window']['start'],v['decode_window']['end']
    start,end=capture_start,capture_end
    completions=[dict(x) for x in c.execute("select start,text from NVTX_EVENTS where text like 'DECODE_OUTPUT_%' order by start")]
    trimmed=False
    if args['steps']>=64 and len(completions)==args['steps']:
        start,end=completions[15]['start'],completions[-17]['start'];trimmed=True
    window=end-start
    tables=set(v['tables']);names={r[0]:r[1] for r in c.execute('select id,value from StringIds')}
    rows=[dict(x) for x in c.execute('select start,end,deviceId,globalPid,correlationId,demangledName,graphNodeId,graphId from CUPTI_ACTIVITY_KIND_KERNEL where start<? and end>? order by start',(end,start))] if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else []
    intervals=union((max(start,r['start']),min(end,r['end'])) for r in rows)
    node_count=sum(r['graphNodeId'] is not None for r in rows)
    is_graph=args['mode']=='graph'
    reliable=bool(rows) and (not is_graph or node_count>0 and '_node_' in path.stem)
    graph_tables=[t for t in tables if 'GRAPH' in t]
    graph_rows=[]
    for t in graph_tables:
        cols=[r[1] for r in c.execute(f'pragma table_info("{t}")')]
        if 'start' in cols and 'end' in cols:
            graph_rows.extend(dict(x,table=t) for x in c.execute(f'select * from "{t}" where start<? and end>?',(end,start)))
    active=sum(b-a for a,b in intervals)
    result=dict(trace=path.stem,args=args,protocol=raw.get('protocol','sync_diagnostic_v1'),
        capture_window_ms=(capture_end-capture_start)/1e6,analysis_start_ns=start,analysis_end_ns=end,
        trimmed_first_last_16_steps=trimmed,analyzed_steps=args['steps']-32 if trimmed else args['steps'],
        window_ms=window/1e6,kernel_count=len(rows),kernel_union_ms=active/1e6 if reliable else None,
        kernel_coverage=active/window if reliable else None,kernel_coverage_reliable=reliable,
        graph_node_count=node_count,graph_table_counts={t:c.execute(f'select count(*) from "{t}"').fetchone()[0] for t in graph_tables},
        graph_envelope_count=len(graph_rows),runtime_api_counts=v['runtime_api_counts'],
        validation=v['validation'],sampling=v.get('sampling'),diagnostics=v['diagnostics'],
        model_step_mean_ms=raw['runs'][0]['mean_step_ms'],step_markers=v['step_markers'],metrics=[])
    # Correlation IDs can prove launches/sync APIs overlap a gap, not that CPU causes every ns.
    runtimes=[dict(x) for x in c.execute('select start,end,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start<? and end>?',(end,start))]
    gaps=[]
    if reliable:
        cursor=start
        for a,b in intervals:
            if a>cursor:gaps.append([cursor,a])
            cursor=max(cursor,b)
        if cursor<end:gaps.append([cursor,end])
    launch_intervals=union((max(start,r['start']),min(end,r['end'])) for r in runtimes if any(x in names[r['nameId']].lower() for x in ['launchkernel','graphlaunch']))
    sync_intervals=union((max(start,r['start']),min(end,r['end'])) for r in runtimes if 'synchronize' in names[r['nameId']].lower())
    gap_launch=sum(overlap_length(launch_intervals,a,b) for a,b in gaps)
    gap_sync=sum(overlap_length(sync_intervals,a,b) for a,b in gaps)
    result.update(gap_ms=(window-active)/1e6 if reliable else None,
        gap_count=len(gaps) if reliable else None,gap_p50_us=percentile([b-a for a,b in gaps],.5)/1000 if gaps else None,
        gap_p95_us=percentile([b-a for a,b in gaps],.95)/1000 if gaps else None,
        gap_overlapping_launch_api_ms=gap_launch/1e6 if reliable else None,
        gap_overlapping_sync_api_ms=gap_sync/1e6 if reliable else None)
    # Per-step union clipped to each actual CPU NVTX decode step.
    steps=[]
    step_ranges=[]
    previous=capture_start
    if completions:
        for s in completions:
            if s['start']>=start and previous>=start and s['start']<=end:
                step_ranges.append(dict(start=previous,end=s['start'],text=s['text']))
            previous=s['start']
    else:
        step_ranges=[dict(x) for x in c.execute("select start,end,text from NVTX_EVENTS where text like 'DECODE_STEP_%' and start>=? and end<=? order by start",(start,end))]
    for s in step_ranges:
        dur=s['end']-s['start'];busy=overlap_length(intervals,s['start'],s['end']) if reliable else None
        steps.append(dict(trace=path.stem,step=s['text'],meaning='Between CPU token-completion notifications' if completions else 'Legacy synchronous engine-step range',start_ns=s['start'],end_ns=s['end'],step_ms=dur/1e6,
            kernel_ms=busy/1e6 if busy is not None else None,gap_ms=(dur-busy)/1e6 if busy is not None else None))
    csv_write(O/'analysis'/f'{path.stem}_steps.csv',steps)
    csv_write(O/'analysis'/f'{path.stem}_kernels.csv',[dict(trace=path.stem,**r) for r in rows])
    csv_write(O/'analysis'/f'{path.stem}_gaps.csv',[dict(trace=path.stem,start_ns=a,end_ns=b,duration_us=(b-a)/1000) for a,b in gaps])
    metrics={}
    samples=[]
    if 'GPU_METRICS' in tables:
        for r in c.execute('select g.timestamp,g.typeId,g.metricId,g.value,m.metricName from GPU_METRICS g join TARGET_INFO_GPU_METRICS m on g.typeId=m.typeId and g.metricId=m.metricId where g.timestamp>=? and g.timestamp<=? order by g.timestamp',(start,end)):
            name=r['metricName'];value=r['value']
            valid=(0<=value<=100) if 'Throughput %' in name else (0<value<10000) if 'Clock Frequency' in name else value>=0
            metrics.setdefault(name,[]).append((r['timestamp'],value,valid))
        starts=[a for a,b in intervals]
        for name,series in metrics.items():
            vals=[value for t,value,valid in series if valid]
            av=[];gv=[]
            for t,value,valid in series:
                j=bisect.bisect_right(starts,t)-1
                act=reliable and j>=0 and t<intervals[j][1]
                if valid and reliable:(av if act else gv).append(value)
                samples.append(dict(trace=path.stem,timestamp_ns=t,metric=name,value=value,valid=valid,
                    kernel_active=int(act) if reliable else None))
            result['metrics'].append(dict(name=name,valid_samples=len(vals),invalid_samples=len(series)-len(vals),
                mean=statistics.mean(vals) if vals else None,p50=percentile(vals,.5),p95=percentile(vals,.95),peak=max(vals) if vals else None,
                active_mean=statistics.mean(av) if av else None,gap_mean=statistics.mean(gv) if gv else None,
                active_samples=len(av),gap_samples=len(gv)))
        csv_write(O/'analysis'/f'{path.stem}_metrics.csv',samples)
    # Compact, deterministic plotting data: activity and metrics over the same 1 ms bins.
    series_out=[];bin_ns=1_000_000
    graph_union=union((r['start'],r['end']) for r in graph_rows)
    metric_bins={}
    for name,series in metrics.items():
        if any(s in name for s in ['SMs Active','Tensor Active','DRAM Read','DRAM Write','SM Issue','Compute Warps in Flight [Throughput']):
            buckets=collections.defaultdict(list)
            for t,val,valid in series:
                if valid:buckets[(t-start)//bin_ns].append(val)
            metric_bins[name]={i:statistics.mean(vals) for i,vals in buckets.items()}
    for a in range(start,end,bin_ns):
        b=min(a+bin_ns,end)
        record=dict(time_ms=(a-start)/1e6,bin_ms=(b-a)/1e6,
            kernel_coverage=overlap_length(intervals,a,b)/(b-a) if reliable else None,
            graph_coverage=overlap_length(graph_union,a,b)/(b-a) if graph_rows else None)
        for name,buckets in metric_bins.items():
            record[name]=buckets.get((a-start)//bin_ns)
        series_out.append(record)
    native=sum('SM120_16x8x64_TN_VS' in names.get(r['demangledName'],'') and 'float_e2m1_t' in names.get(r['demangledName'],'') for r in rows)
    result['native_sm120_e2m1_kernel_instances']=native
    result['engine_core_pid_mapping_match']=all(d['pid']==raw['pid'] for d in v['cuda_devices'])
    result['kernel_names']=[dict(name=key,count=count) for key,count in collections.Counter(names.get(r['demangledName'],'') for r in rows).most_common()]
    result['plot']=dict(series=series_out,
        short_kernels=[[max(a,start)-start,min(b,end)-start] for a,b in intervals if a<start+25_000_000],
        short_graphs=[[max(r['start'],start)-start,min(r['end'],end)-start] for r in graph_rows if r['start']<start+25_000_000],
        short_gaps=[[a-start,b-start] for a,b in gaps if a<start+25_000_000])
    dest=O/'analysis'/f'{path.stem}_summary.json';dest.write_text(json.dumps(result))
    v['validation']['clock_alignment_status']='shared SessionNs time source, PCI/UUID mapping, overlapping NVTX/CUDA ranges; no timestamp adjustment'
    (O/'analysis'/f'{path.stem}_validation.json').write_text(json.dumps(v,indent=2))
    c.close()
    print(path.stem,round(result['window_ms'],2),'ms','coverage',result['kernel_coverage'],'samples',v.get('sampling',{}).get('inside_window'))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--file');a=ap.parse_args()
    paths=[Path(a.file)] if a.file else sorted((O/'sqlite').glob('*.sqlite'))
    results=[analyze_one(p) for p in paths]
    (O/'analysis/timestamp_validation.json').write_text(json.dumps({r['trace']:r['validation'] for r in results},indent=2))
    csv_write(O/'analysis/metric_summary.csv',[dict(trace=r['trace'],**m) for r in results for m in r['metrics']])
    baseline=[];diagnostic=[]
    for f in sorted((O/'raw').glob('*_baseline_*.json')):
        d=json.loads(f.read_text())
        for r in d['runs']:
            row=dict(file=f.name,precision=d['args']['precision'],batch=d['args']['batch'],mode=d['args']['mode'],
                trial=r['trial'],decode_steps=r['decode_steps'],decode_wall_ms=r['decode_wall_ms'],
                mean_step_ms=r['mean_step_ms'],p50_step_ms=r['p50_step_ms'],p95_step_ms=r['p95_step_ms'],
                output_tokens_per_s=r['output_tokens_per_s'],actual_batch=r['actual_batches'][0],graph_replays=r['graph_replays'],
                cuda_reserved_bytes=d['cuda_reserved_bytes'],pid=d['pid'])
            (baseline if d.get('protocol')=='default_scheduler_v2' else diagnostic).append(row)
    csv_write(O/'baseline.csv',baseline)
    csv_write(O/'analysis/baseline_sync_diagnostic.csv',diagnostic)
    print('BASELINE_ROWS',len(baseline))

if __name__=='__main__':main()
