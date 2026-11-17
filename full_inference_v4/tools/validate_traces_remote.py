"""CPU validation of completed traces. No profiler/model is launched."""
from pathlib import Path
import sys,json,collections,statistics,re,sqlite3,math
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from analyze_timeline import analyze
from analyze_resources import read_samples,summarize_samples
from common import csv_write

def fast_summarize(samples,windows):
 from common import union,duration,percentile
 grouped=collections.defaultdict(list)
 for s in samples:
  if math.isfinite(s['value']):grouped[(s['typeId'],s['metricId'],s['name'])].append(s)
 result=[]
 for (type_id,metric_id,name),ss in grouped.items():
  ss.sort(key=lambda s:s['timestamp'])
  positive=[b['timestamp']-a['timestamp'] for a,b in zip(ss,ss[1:]) if b['timestamp']>a['timestamp']]
  cadence=statistics.median(positive) if positive else None
  for scope,intervals in windows.items():
   merged=union(intervals);j=0;weight=total=0;values=[]
   for a,b in zip(ss,ss[1:]):
    lo,hi=a['timestamp'],b['timestamp']
    if not cadence or hi-lo>1.5*cadence:continue
    if ('Frequency' in name or 'Clock' in name) and a.get('value_status')!='VALIDATED_UNIT_RANGE':continue
    while j<len(merged) and merged[j][1]<=lo:j+=1
    k=j;dt=0
    while k<len(merged) and merged[k][0]<hi:
     dt+=max(0,min(hi,merged[k][1])-max(lo,merged[k][0]));k+=1
    if dt:total+=a['value']*dt;weight+=dt;values.append(a['value'])
   result.append(dict(typeId=type_id,metricId=metric_id,metric=name,stage=scope,
     time_weighted_mean=total/weight if weight else 'UNAVAILABLE',p50_sample=percentile(values,.5),p95_sample=percentile(values,.95),
     valid_interval_ns=weight,scope_window_ns=duration(merged),interval_coverage=weight/duration(merged) if duration(merged) else 0,
     sample_intervals=len(values),measurement_tool='NSYS GPU Metrics',
     measurement_scope='device-level samples conditioned on labelled intervals; not per-kernel counters',
     measurement_status='UNVERIFIED_UNTIL_TIMESTAMP_AND_SOURCE_AUDIT'))
 return result

all_results=[]
for path in sorted((root/'sqlite').glob('gate2_*.sqlite')):
 existing=root/'analysis'/(path.stem+'_validation.json')
 if existing.exists():
  all_results.append(json.loads(existing.read_text()));continue
 summary,activities=analyze(path,root/'analysis')
 db=sqlite3.connect(path);db.row_factory=sqlite3.Row
 tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 ranges=summary['cpu_full_range'];raw=json.loads((root/'raw'/(path.stem+'.json')).read_text())
 run=raw['runs'][0];cfg=raw['config']
 assert summary['gpu_identity_status']=='VERIFIED_METADATA'
 assert len(summary['context_ids'])==1 and len(summary['device_ids'])==1 and len(summary['global_pids'])==1
 severe=[d for d in summary['diagnostics'] if d.get('severity',0)>=3 or re.search('overflow|missing data|data loss',d.get('text',''),re.I)]
 # Some NSYS capture-boundary warnings contain "might"; retained for completeness audit.
 assert not severe,('NSYS_DIAGNOSTICS',severe)
 sources=[dict(r) for r in db.execute('SELECT * FROM GENERIC_EVENT_SOURCES')] if 'GENERIC_EVENT_SOURCES' in tables else []
 samples=read_samples(path)
 sampling=None
 if samples:
  assert all(s.get('timeSourceId')==3 for s in sources),'non-SessionNs metric source'
  timestamps=sorted({s['timestamp'] for s in samples});dt=[b-a for a,b in zip(timestamps,timestamps[1:])]
  median=statistics.median(dt)
  sampling=dict(samples=len(timestamps),dt_median_ns=median,dt_p95_ns=sorted(dt)[int(.95*(len(dt)-1))],
   gap_count=sum(d>1.5*median for d in dt),min_ns=timestamps[0],max_ns=timestamps[-1],
   samples_inside_cpu_full=sum(ranges['start']<=t<=ranges['end'] for t in timestamps))
  assert sampling['samples_inside_cpu_full']>0,'no metrics in full window'
  assert sampling['gap_count']==0,'metric sample gaps'
  windows={'FULL_CPU_REQUEST_WINDOW':[(ranges['start'],ranges['end'])]}
  for a in activities:
   if a['stage']!='UNKNOWN':windows.setdefault(a['stage'],[]).append((a['start'],a['end']))
  metric_summary=fast_summarize(samples,windows)
  csv_write(root/'analysis'/(path.stem+'_resource_summary.csv'),metric_summary)
 else:metric_summary=[]
 names=collections.Counter(a['kernel_name'] for a in activities if a['activity_kind']=='KERNEL')
 unknown=collections.Counter(a['stage_status'] for a in activities if a['stage']=='UNKNOWN')
 result=dict(file=path.name,profile=raw['profile'],experiment=cfg['experiment'],precision=cfg['precision'],
   gpu_identity=summary['gpu_identity_status'],kernel_count=summary['kernel_count'],graph_count=summary['graph_envelope_count'],
   unknown_stages=summary['stage_assignment_unknown_count'],unknown_reasons=dict(unknown),kernel_families=names.most_common(),
   sampling=sampling,source_timebase_metadata=sources,diagnostics=summary['diagnostics'],
   full_ms=run['full_model_wall_ms'],replays=len(run['graph_replays']),stage_scope='activity envelopes, not per-kernel utilization')
 (root/'analysis'/(path.stem+'_validation.json')).write_text(json.dumps(result,indent=2))
 all_results.append(result)
 print('TRACE_VALIDATION',path.stem,'kernels',result['kernel_count'],'graphs',result['graph_count'],'unknown',result['unknown_stages'],'sampling',sampling)
 db.close()
(root/'analysis/gate2_trace_validation.json').write_text(json.dumps(all_results,indent=2))
