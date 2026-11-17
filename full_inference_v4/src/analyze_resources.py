"""Device samples retain device scope, even when conditioned on a stage envelope."""
from collections import defaultdict
import math
import sqlite3
import statistics
from common import duration,union,intersection,percentile


def summarize_samples(samples,windows):
    """Piecewise-constant integration, timestamps in ns, no inferred offsets."""
    grouped=defaultdict(list)
    for s in samples:
        if math.isfinite(s['value']):grouped[(s['typeId'],s['metricId'],s['name'])].append(s)
    out=[]
    for (type_id,metric_id,name),ss in grouped.items():
        ss.sort(key=lambda s:s['timestamp'])
        positive=[b['timestamp']-a['timestamp'] for a,b in zip(ss,ss[1:]) if b['timestamp']>a['timestamp']]
        cadence=statistics.median(positive) if positive else None
        for scope,intervals in windows.items():
            weight=total=0;values=[]
            for a,b in zip(ss,ss[1:]):
                if not cadence or b['timestamp']-a['timestamp']>1.5*cadence:continue
                dt=duration(intersection([(a['timestamp'],b['timestamp'])],intervals))
                if dt:
                    # Invalid frequency exports are not repaired or reinterpreted.
                    if ('Frequency' in name or 'Clock' in name) and a.get('value_status')!='VALIDATED_UNIT_RANGE':continue
                    total+=a['value']*dt;weight+=dt;values.append(a['value'])
            out.append(dict(typeId=type_id,metricId=metric_id,metric=name,stage=scope,
                            time_weighted_mean=total/weight if weight else 'UNAVAILABLE',
                            p50_sample=percentile(values,.5),p95_sample=percentile(values,.95),
                            valid_interval_ns=weight,scope_window_ns=duration(intervals),
                            interval_coverage=weight/duration(intervals) if duration(intervals) else 0,
                            sample_intervals=len(values),measurement_tool='NSYS GPU Metrics',
                            measurement_scope='device-level samples conditioned on labelled intervals; not per-kernel counters',
                            measurement_status='UNVERIFIED_UNTIL_TIMESTAMP_AND_SOURCE_AUDIT'))
    return out


def read_samples(path):
    c=sqlite3.connect(str(path));c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'GPU_METRICS','TARGET_INFO_GPU_METRICS'}<=tables:return []
    fields={r[1] for r in c.execute('PRAGMA table_info(GPU_METRICS)')}
    if not {'timestamp','typeId','metricId','value'}<=fields:raise ValueError('unsupported metric schema')
    rows=[dict(r) for r in c.execute('SELECT g.timestamp,g.typeId,g.metricId,g.value,m.metricName AS name FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m ON g.typeId=m.typeId AND g.metricId=m.metricId ORDER BY g.timestamp')]
    c.close();return rows


if __name__=='__main__':
    import argparse
    from common import load,csv_write
    p=argparse.ArgumentParser();p.add_argument('--sqlite',required=True);p.add_argument('--activities',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    rows=load(a.activities);windows={}
    for r in rows:
        if r['stage']!='UNKNOWN':windows.setdefault(r['stage'],[]).append((r['start'],r['end']))
    if rows:windows['FULL_GPU_ACTIVITY_ENVELOPE']=[(min(r['start'] for r in rows),max(r['end'] for r in rows))]
    # Phase windows can overlap; they are conditional views, never additive budgets.
    result=summarize_samples(read_samples(a.sqlite),windows)
    if result:csv_write(a.output,result)
    else:print('UNAVAILABLE: no GPU Metrics samples; no zero-filled counters written')
