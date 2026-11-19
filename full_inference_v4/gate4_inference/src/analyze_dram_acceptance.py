"""Match isolated DRAM directions to device samples wholly inside their GPU kernels."""
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys


def measure(db, data, guard):
    if (data['profile']!='P1' or data['status']!='STANDALONE_FULL_OUTPUTS_VERIFIED'
            or guard['status']!='COMPLETED' or '--gpu-metrics-frequency=10000' not in guard['command']):
        raise ValueError('expected supervised P1')
    dev, = db.execute('select g.uuid,d.uuid,g.id,d.cudaId,d.pid from TARGET_INFO_GPU g join TARGET_INFO_CUDA_DEVICE d on g.id=d.gpuId')
    if dev[0]!=dev[1] or dev[0]!=data['gpu_uuid'].removeprefix('GPU-') or dev[3]!=0 or dev[4] not in guard['known_child_pids']:
        raise ValueError('GPU identity changed')
    source,=db.execute('select distinct m.sourceId,s.timeSourceId from TARGET_INFO_GPU_METRICS m join GENERIC_EVENT_SOURCES s using(sourceId)')
    if source[1]!=3 or db.execute('select count(*) from TARGET_INFO_GPU').fetchone()[0]!=1:
        raise ValueError('unverified clock or device source')
    diagnostics=[dict(r) for r in db.execute('select severity,text from DIAGNOSTIC_EVENT')]
    if any(r['severity']>=3 or re.search('overflow|missing data|data loss',r['text'],re.I) for r in diagnostics):
        raise ValueError('incomplete trace')
    metrics=defaultdict(dict)
    for t,name,v in db.execute('select g.timestamp,m.metricName,g.value from GPU_METRICS g join TARGET_INFO_GPU_METRICS m using(typeId,metricId)'):
        if name.startswith(('DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]')):
            metrics[t][name.split()[1].lower()]=v
    times=sorted(metrics)
    cadence=statistics.median(b-a for a,b in zip(times,times[1:]))
    if not 95000<cadence<105000:
        raise ValueError('unexpected sampling cadence')
    def interior(windows):
        samples=[]
        for a,b in windows:
            for i in range(max(1,bisect_left(times,a)),min(len(times)-1,bisect_left(times,b))):
                if times[i-1]>=a and times[i+1]<=b and times[i+1]-times[i-1]<=2.5*cadence:
                    samples.append(metrics[times[i]])
        return dict(samples=len(samples),**{d:statistics.mean(s[d] for s in samples) if samples else None for d in ('read','write')})
    kernels=defaultdict(list)
    for name,a,b in db.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id'):
        kernels[name].append((a,b))
    idle,=db.execute("select start,end from NVTX_EVENTS where text='DRAM_IDLE_CONTROL'")
    idle_window=(idle[0]+2_000_000,idle[1]-2_000_000)
    if any(a<idle_window[1] and b>idle_window[0] for windows in kernels.values() for a,b in windows):
        raise ValueError('background control overlaps CUDA kernel')
    base=interior([idle_window])
    if base['samples']<20:
        raise ValueError('missing background comparison')
    rows=[]
    for spec in data['results']:
        windows=kernels[spec['symbol']]
        if len(windows)!=spec.get('launches',9) or spec['region_bytes']<4*data['l2_bytes'] or not spec['correctness_passed']:
            raise ValueError('unexpected work or kernel count')
        observed=interior(windows)
        directions={'read':('read',),'write':('write',),'copy':('read','write')}[spec['operation']]
        passed=observed['samples']>=5 and all(observed[d]>base[d]+1 for d in directions)
        rows.append(dict(symbol=spec['symbol'], kernel_calls=len(windows), interior_samples=observed['samples'],
                         read_pct=observed['read'],write_pct=observed['write'], direction_detected=passed,
                         physical_dram_bytes='UNAVAILABLE', duration_median_us=statistics.median((b-a)/1000 for a,b in windows)))
    return dict(run_id=data['run_id'],uuid=dev[0],cubin_sha256=data['cubin_sha256'],
                cadence_ns=cadence,idle=base,rows=rows,diagnostics=diagnostics)


if __name__=='__main__':
    root,run=Path(sys.argv[1]),sys.argv[2]
    with sqlite3.connect((root/'sqlite'/(run+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        result=measure(db,json.loads((root/'raw'/(run+'.json')).read_bytes()),
                       json.loads((root/'logs'/(run+'_supervision.json')).read_bytes()))
        print(json.dumps(result))
