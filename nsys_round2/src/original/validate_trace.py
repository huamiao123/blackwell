from pathlib import Path
import sqlite3,json,statistics,argparse,math

def validate(path):
    c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    result={'file':str(path),'tables':sorted(tables)}
    result['schema']={t:[dict(x) for x in c.execute(f'pragma table_info("{t}")')] for t in tables if t in ['GPU_METRICS','TARGET_INFO_GPU_METRICS','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL','TARGET_INFO_GPU','TARGET_INFO_CUDA_DEVICE'] or 'GRAPH' in t}
    windows=[dict(x) for x in c.execute("select start,end,text,globalTid from NVTX_EVENTS where text='ROUND2_DECODE_WINDOW'")]
    assert len(windows)==1 and windows[0]['end']>windows[0]['start'],windows
    start,end=windows[0]['start'],windows[0]['end']
    result['decode_window']=windows[0]
    result['gpu_inventory']=[dict(x) for x in c.execute('select * from TARGET_INFO_GPU')]
    result['cuda_devices']=[dict(x) for x in c.execute('select * from TARGET_INFO_CUDA_DEVICE')]
    result['metric_sources']=[dict(x) for x in c.execute('select * from GENERIC_EVENT_SOURCES')] if 'GENERIC_EVENT_SOURCES' in tables else []
    result['metric_source_names']=[dict(x) for x in c.execute('select g.typeId,g.sourceId,s.value as name from GENERIC_EVENT_TYPES g join StringIds s on s.id=g.nameId')] if 'GENERIC_EVENT_TYPES' in tables else []
    result['diagnostics']=[dict(x) for x in c.execute('select * from DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    result['kernels']=[dict(x) for x in c.execute('select count(*) as count,min(start) as min_start,max(end) as max_end,deviceId,globalPid from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId,globalPid')] if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else []
    result['step_markers']=c.execute("select count(*) from NVTX_EVENTS where text like 'DECODE_STEP_%' and start>=? and end<=?",(start,end)).fetchone()[0]
    result['runtime_api_counts']=[dict(x) for x in c.execute('select s.value as name,count(*) as count from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.end<=? group by s.value',(start,end))]
    if 'GPU_METRICS' in tables:
        ts=[x[0] for x in c.execute('select distinct timestamp from GPU_METRICS order by timestamp')]
        dt=sorted(b-a for a,b in zip(ts,ts[1:]))
        median_dt=statistics.median(dt)
        result['sampling']=dict(count=len(ts),min_ns=ts[0],max_ns=ts[-1],dt_p50_ns=median_dt,dt_p95_ns=dt[int(.95*(len(dt)-1))],dt_max_ns=max(dt),inside_window=sum(start<=t<=end for t in ts),intervals_gt_1_5x_median=sum(v>1.5*median_dt for v in dt))
        result['metrics']=[dict(x) for x in c.execute('select m.metricName as name,count(*) as count,min(g.value) as min,max(g.value) as max,avg(g.value) as mean from GPU_METRICS g join TARGET_INFO_GPU_METRICS m on g.typeId=m.typeId and g.metricId=m.metricId where g.timestamp>=? and g.timestamp<=? group by m.metricName',(start,end))]
    # Identity comes from the device-help PCI address + GPU UUID inventory + CUDA UUID.
    inventory=result['gpu_inventory'];devices=result['cuda_devices']
    identity_ok=(len(inventory)==1 and inventory[0]['busLocation']=='0000:27:00.0' and
        all(d['uuid']==inventory[0]['uuid'] for d in devices))
    source_session_ns=all(r['timeSourceId']==3 for r in result['metric_sources']) if result['metric_sources'] else None
    result['activity_boundary_check']=[]
    if 'GPU_METRICS' in tables:
        result['activity_boundary_check']=[dict(x) for x in c.execute('''
            select m.metricName as metric,
                   case when g.timestamp<? then 'before' when g.timestamp<=? then 'decode' else 'after' end as phase,
                   count(*) as samples,avg(g.value) as mean
            from GPU_METRICS g join TARGET_INFO_GPU_METRICS m on g.typeId=m.typeId and g.metricId=m.metricId
            where m.metricName in ('SMs Active [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]')
            group by m.metricName,phase''',(start,end))]
    result['validation']=dict(gpu_identity_match=identity_ok,metrics_source_session_ns=source_session_ns,
        timestamps_used='GPU_METRICS.timestamp / CUPTI start,end / NVTX start,end; no offsets applied',
        ranges_overlap=('sampling' not in result or result['sampling']['min_ns']<=end and result['sampling']['max_ns']>=start),
        clock_alignment_status='schema and range verified; inspect activity/idle correspondence in figures',
        note='Graph envelopes are never used as kernel coverage. Capture-boundary warnings retained, not suppressed.')
    c.close()
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('file');a=ap.parse_args()
    path=Path(a.file);result=validate(path)
    dest=path.parent.parent/'analysis'/f'{path.stem}_validation.json'
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ['decode_window','step_markers','sampling','validation','kernels']},indent=2))
    print('WARNINGS',[(r['severity'],r['text']) for r in result['diagnostics'] if r['severity']>=2])
