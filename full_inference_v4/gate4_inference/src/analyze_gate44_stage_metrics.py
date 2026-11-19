"""Compute device-level GB20x metrics on correlated E2 GPU stage envelopes."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys


def weighted(samples, intervals, cadence):
    ts = [r[0] for r in samples]
    total, weight, values = 0., 0, []
    for start, end in intervals:
        for i in range(max(0,bisect_right(ts,start)-1), min(len(ts)-1,bisect_left(ts,end))):
            dt = min(ts[i+1],end) - max(ts[i],start)
            if dt <= 0 or ts[i+1]-ts[i] > 1.5*cadence:
                continue
            value = samples[i][1]
            if not 0 <= value <= 100:
                raise ValueError('out-of-range percent metric')
            total += value*dt
            weight += dt
            values.append(value)
    size = sum(b-a for a,b in intervals)
    if not values or weight/size < .98:
        raise ValueError('incomplete metric window')
    return dict(mean_pct=total/weight, median_pct=statistics.median(values),
                sample_intervals=len(values),interval_coverage=weight/size,window_ns=size)


def measure(db, run, guard):
    if (run['profile']!='P1' or run['invocation_status']!='COMPLETED' or guard['status']!='COMPLETED'
            or '--gpu-metrics-frequency=10000' not in guard['command'] or '--gpu-metrics-devices=0' not in guard['command']):
        raise ValueError('incomplete 10 kHz P1 diagnostic')
    dev, = db.execute('select d.cudaId,d.pid,d.uuid,g.uuid as physical from TARGET_INFO_CUDA_DEVICE d '
                      'join TARGET_INFO_GPU g on g.id=d.gpuId')
    if (dev['pid'] not in guard['known_child_pids'] or dev['uuid']!=dev['physical']
            or dev['cudaId']!=0 or dev['uuid']!=run['gpu_uuid'].removeprefix('GPU-')
            or db.execute('select count(*) from TARGET_INFO_GPU').fetchone()[0]!=1):
        raise ValueError('ambiguous GPU identity')
    sources=list(db.execute('select distinct m.sourceId,s.timeSourceId from TARGET_INFO_GPU_METRICS m '
                            'left join GENERIC_EVENT_SOURCES s on s.sourceId=m.sourceId'))
    if len(sources)!=1 or sources[0]['timeSourceId']!=3:
        raise ValueError('unverified metric source or clock')
    diagnostics=[dict(d) for d in db.execute('select severity,text from DIAGNOSTIC_EVENT')]
    if any(d['severity']>=3 or re.search('buffer overflow|missing data|data loss',d['text'],re.I) for d in diagnostics):
        raise ValueError('incomplete profiler data')
    names={r[0]:r[1] for r in db.execute('select id,value from StringIds')}
    labels=[(r['text'] or names.get(r['textId'],''),r) for r in db.execute('select * from NVTX_EVENTS where end>start')]
    full,=[(r['start'],r['end']) for n,r in labels if n=='FULL_INFERENCE']
    calls=defaultdict(list)
    for r in db.execute('select globalTid,start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        calls[r['globalTid']].append(r)
    graphs={r['correlationId']:r for r in db.execute('select * from CUPTI_ACTIVITY_KIND_GRAPH_TRACE')}
    if len(graphs)!=256:
        raise ValueError('expected 256 distinct Decode graph launches')
    contexts={(r['globalPid'],r['contextId'],r['deviceId']) for r in graphs.values()}
    if len(contexts)!=1:
        raise ValueError('multiple namespaces')
    kernels=defaultdict(list)
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'):
        if (r['globalPid'],r['contextId'],r['deviceId']) not in contexts:
            raise ValueError('CUDA namespaces differ')
        kernels[r['correlationId']].append(r)
    decode={}
    prefill=[]
    for name,mark in labels:
        is_prefill = name=='ITERATION_DISPATCH|0|PREFILL'
        if not is_prefill and not name.startswith('GRAPH_REPLAY|'):
            continue
        correlated=[c['correlationId'] for c in calls[mark['globalTid']]
                    if mark['start']<=c['start'] and c['end']<=mark['end']]
        if is_prefill:
            prefill.extend((k['start'],k['end']) for c in correlated for k in kernels[c]
                           if names[k['shortName']]!='tensor_light')
        else:
            _,_,index=name.split('|')
            graph,=[graphs[c] for c in correlated if c in graphs]
            if int(index) in decode:
                raise ValueError('duplicate replay')
            decode[int(index)]=(graph['start'],graph['end'])
    if set(decode)!=set(range(1,257)) or not prefill:
        raise ValueError('missing stage mapping')
    ordered=sorted(decode.values())
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('overlapping decode windows')
    windows=dict(full_cpu_request=[full], prefill_gpu_span=[(min(a for a,b in prefill),max(b for a,b in prefill))],
                 decode_33_64_graphs=[decode[i] for i in range(33,65)], decode_all_graphs=ordered)
    samples=defaultdict(list)
    for t,n,v in db.execute('select g.timestamp,m.metricName,g.value from GPU_METRICS g '
                             'join TARGET_INFO_GPU_METRICS m using(typeId,metricId) order by g.timestamp'):
        if n.endswith(' [Throughput %]') and n.split(' [')[0] in ('SMs Active','SM Issue','Tensor Active','DRAM Read Bandwidth','DRAM Write Bandwidth',
                                'Compute Warps in Flight'):
            samples[n].append((t,v))
    if len(samples)!=6:
        raise ValueError(f'unexpected resource set {list(samples)}')
    rows=[]
    for n,ss in samples.items():
        delta=[b[0]-a[0] for a,b in zip(ss,ss[1:])]
        cadence=statistics.median(delta)
        if not 95000<=cadence<=105000 or min(delta)<=0:
            raise ValueError('invalid cadence')
        for stage,intervals in windows.items():
            rows.append(dict(metric=n,stage=stage,median_interval_ns=cadence,**weighted(ss,intervals,cadence)))
    return dict(run_id=run['run_id'], precision=run['config']['precision'],condition=run['gate44']['condition'],
                uuid=dev['uuid'],source_id=sources[0]['sourceId'],diagnostics=diagnostics,rows=rows,
                probe_calls=run['gate44']['trials'][0]['completed_probes'],gpu_graphs=len(decode),
                sample_scope='device averages over trace-specific windows, no per-probe counters')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('run')
    a=parser.parse_args()
    def load(folder,suffix):
        return json.loads((a.root/folder/(a.run+suffix)).read_text())
    with sqlite3.connect((a.root/'sqlite'/(a.run+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        print(json.dumps(measure(db,load('raw','.json'),load('logs','_supervision.json'))))
