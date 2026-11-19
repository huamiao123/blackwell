"""Baseline device counters at Decode graph entry in a frozen A-alone P1 trace."""
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys
from analyze_gate44_stage_metrics import measure


def graph_entry(db, run, guard):
    checked = measure(db,run,guard)
    if checked['condition']!='a_alone' or checked['probe_calls']!=0:
        raise ValueError('control must contain no auxiliary activity')
    graphs={r['correlationId']:r for r in db.execute('select * from CUPTI_ACTIVITY_KIND_GRAPH_TRACE')}
    calls=defaultdict(list)
    for c in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        calls[c['globalTid']].append(c)
    intervals={}
    for mark in db.execute("select * from NVTX_EVENTS where end>start and text like 'GRAPH_REPLAY|%'"):
        iteration=int(mark['text'].split('|')[2])
        if iteration not in range(33,65):
            continue
        graph,=[graphs[c['correlationId']] for c in calls[mark['globalTid']] if mark['start']<=c['start']
                and c['end']<=mark['end'] and c['correlationId'] in graphs]
        if iteration in intervals:
            raise ValueError('duplicate Decode iteration')
        intervals[iteration]=(graph['start'],graph['end'])
    if set(intervals)!=set(range(33,65)):
        raise ValueError('missing Decode iteration')
    values=defaultdict(dict)
    for t,n,v in db.execute('select g.timestamp,m.metricName,g.value from GPU_METRICS g join TARGET_INFO_GPU_METRICS m using(typeId,metricId)'):
        if n in ('DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]','SMs Active [Throughput %]','Tensor Active [Throughput %]'):
            values[t][n.split(' [')[0]]=v
    times=sorted(values)
    samples=[]
    for a,b in intervals.values():
        if b-a<=500000:
            raise ValueError('graph shorter than control window')
        for i in range(max(1,bisect_left(times,a)),min(len(times)-1,bisect_left(times,a+500000))):
            if times[i-1]>=a and times[i+1]<=a+500000 and times[i+1]-times[i-1]<250000:
                samples.append(values[times[i]])
    if len(samples)<32:
        raise ValueError('insufficient interior samples')
    return dict(run_id=run['run_id'],precision=checked['precision'],phase='first 500 us of Decode replays 33-64',
                probe_calls=0,samples=len(samples),device_pct={n:statistics.mean(s[n] for s in samples) for n in samples[0]},
                scope='archived A-alone P1 device control, independent of the new P2 run')


if __name__=='__main__':
    root,run=Path(sys.argv[1]),sys.argv[2]
    with sqlite3.connect((root/'sqlite'/(run+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        print(json.dumps(graph_entry(db,json.loads((root/'raw'/(run+'.json')).read_bytes()),
                                    json.loads((root/'logs'/(run+'_supervision.json')).read_bytes()))))
