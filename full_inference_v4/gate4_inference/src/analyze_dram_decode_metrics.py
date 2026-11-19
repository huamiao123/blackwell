"""Analyze 10 kHz samples inside the streamed read and the measured Decode replay."""
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys

from analyze_gate44_stage_metrics import measure


def conditional(db,run,guard):
    validated=measure(db,run,guard)
    if validated['probe_calls']!=32 or run['gate44']['probe']['symbol']!='dram_read_light':
        raise ValueError('wrong probe')
    rts=defaultdict(list)
    for rt in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        rts[rt['globalTid']].append(rt)
    graphs={r['correlationId']:r for r in db.execute('select * from CUPTI_ACTIVITY_KIND_GRAPH_TRACE')}
    probes={r['correlationId']:r for r in db.execute("select k.* from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='dram_read_light'")}
    if len(probes)!=32:
        raise ValueError('unexpected streamed read count')
    spans=defaultdict(dict)
    for r in db.execute("select * from NVTX_EVENTS where end>start and (text like 'GRAPH_REPLAY|%' or text like 'G44_B|%')"):
        label=r['text'].split('|')
        index=int(label[2])
        if index not in range(33,65):
            continue
        which='probe' if label[0]=='G44_B' else 'graph'
        source=probes if which=='probe' else graphs
        kernel,=[source[c['correlationId']] for c in rts[r['globalTid']] if r['start']<=c['start'] and c['end']<=r['end']
                 and c['correlationId'] in source]
        if which in spans[index]:
            raise ValueError('duplicate stage label')
        spans[index][which]=kernel
    controls=defaultdict(list)
    durations=[]
    windows=[]
    for i in range(33,65):
        g,b=spans[i]['graph'],spans[i]['probe']
        if any(b[k]!=g[k] for k in ('globalPid','deviceId','contextId')) or b['streamId']==g['streamId']:
            raise ValueError('wrong GPU execution namespace')
        common=(max(b['start'],g['start']),min(b['end'],g['end']))
        parallel=validated['condition']=='parallel'
        if (common[1]>common[0])!=parallel:
            raise ValueError('unexpected graph/probe relationship')
        controls['probe_inside_graph' if parallel else 'probe_before_graph'].append(common if parallel else (b['start'],b['end']))
        if g['end']-g['start']<500000:
            raise ValueError('short graph')
        controls['graph_first_500us'].append((g['start'],g['start']+500000))
        durations.append((b['end']-b['start'])/1000)
        windows.append(dict(iteration=i,graph_start=g['start'],graph_end=g['end'],probe_start=b['start'],probe_end=b['end']))
    metrics=defaultdict(dict)
    for t,n,v in db.execute('select g.timestamp,m.metricName,g.value from GPU_METRICS g join TARGET_INFO_GPU_METRICS m using(typeId,metricId)'):
        if n in ('DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]','SMs Active [Throughput %]','Tensor Active [Throughput %]'):
            metrics[t][n.split(' [')[0]]=v
    times=sorted(metrics)
    rows=[]
    for label,bounds in controls.items():
        samples=[]
        for a,b in bounds:
            for i in range(max(1,bisect_left(times,a)),min(len(times)-1,bisect_left(times,b))):
                if times[i-1]>=a and times[i+1]<=b and times[i+1]-times[i-1]<250000:
                    samples.append(metrics[times[i]])
        if len(samples)<32:
            raise ValueError('insufficient interior samples')
        for n in samples[0]:
            rows.append(dict(window=label,metric=n,mean_pct=statistics.mean(s[n] for s in samples),samples=len(samples)))
    return dict(run_id=run['run_id'],precision=validated['precision'],condition=validated['condition'],uuid=validated['uuid'],
                conditional_device_metrics=rows,probe_kernel_us=durations,windows=windows,stage_metrics=validated['rows'])


if __name__=='__main__':
    path,run=Path(sys.argv[1]),sys.argv[2]
    with sqlite3.connect((path/'sqlite'/(run+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        print(json.dumps(conditional(c,json.loads((path/'raw'/(run+'.json')).read_bytes()),
                                    json.loads((path/'logs'/(run+'_supervision.json')).read_bytes()))))
