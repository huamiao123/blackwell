"""CPU SQLite analysis. Assignment uses launch correlation, never CPU time overlap with kernels."""
import argparse
from collections import defaultdict
from pathlib import Path
import sqlite3
from common import ROOT,load,save,csv_write,union,duration,file_hash


def columns(c,table):
    return {r[1] for r in c.execute(f'PRAGMA table_info("{table}")')}


def labels(c):
    ss={r['id']:r['value'] for r in c.execute('SELECT * FROM StringIds')}
    events=[]
    for r in c.execute('SELECT * FROM NVTX_EVENTS'):
        r=dict(r);r['label']=r.get('text') or ss.get(r.get('textId'),'');events.append(r)
    return ss,events


def analyze(path,out,expected_bus='0000:27:00.0'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro',uri=True);c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ['NVTX_EVENTS','StringIds','CUPTI_ACTIVITY_KIND_RUNTIME']:
        if name not in tables:raise ValueError('missing '+name)
    ss,events=labels(c)
    full=[r for r in events if r['label']=='FULL_INFERENCE' and r.get('end',0)>r['start']]
    if len(full)!=1:raise ValueError('expected one complete full-inference range')
    start,end=full[0]['start'],full[0]['end']
    dispatch=[r for r in events if r['label'].startswith('ITERATION_DISPATCH|') and r.get('end',0)>r['start']]
    by_corr=defaultdict(list)
    runtimes=[dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    for r in runtimes:by_corr[r['correlationId']].append(r)
    rows=[]
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','KERNEL'),('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','GRAPH_ENVELOPE')]:
        if table not in tables:continue
        if not {'start','end','correlationId'}<=columns(c,table):raise ValueError('unsupported activity schema')
        for rr in c.execute(f'SELECT * FROM {table}'):
            r=dict(rr);r['activity_kind']=kind
            r['kernel_name']=ss.get(r.get('demangledName'),ss.get(r.get('shortName'),'UNAVAILABLE'))
            r['stage']='UNKNOWN';r['scheduler_iteration']='UNKNOWN';r['stage_status']='UNVERIFIED'
            candidates=by_corr.get(r['correlationId'],[])
            if len(candidates)==1:
                api=candidates[0]
                matches=[n for n in dispatch if n.get('globalTid')==api.get('globalTid') and
                         n['start']<=api['start'] and api['end']<=n['end']]
                if len(matches)==1:
                    _,iteration,stage=matches[0]['label'].split('|')
                    r.update(scheduler_iteration=int(iteration) if iteration.isdigit() else 'UNKNOWN',
                             stage=stage,stage_status='CORRELATION_MATCHED_PENDING_RUNTIME_AUDIT')
            rows.append(r)
    kernels=[r for r in rows if r['activity_kind']=='KERNEL']
    graphs=[r for r in rows if r['activity_kind']=='GRAPH_ENVELOPE']
    if not rows:raise ValueError('EMPTY_TRACE')
    inventory=[dict(r) for r in c.execute('SELECT * FROM TARGET_INFO_GPU')] if 'TARGET_INFO_GPU' in tables else []
    devices=[dict(r) for r in c.execute('SELECT * FROM TARGET_INFO_CUDA_DEVICE')] if 'TARGET_INFO_CUDA_DEVICE' in tables else []
    identity=bool(inventory and devices) and len(inventory)==1 and inventory[0].get('busLocation')==expected_bus and all(
        r.get('uuid')==inventory[0].get('uuid') for r in devices)
    pids={r.get('globalPid') for r in rows if r.get('globalPid') is not None}
    contexts={r.get('contextId') for r in rows if r.get('contextId') is not None}
    gpu_ids={r.get('deviceId') for r in rows if r.get('deviceId') is not None}
    if len(pids)>1 or len(contexts)>1 or len(gpu_ids)>1:
        # Correlation alone is not accepted across ambiguous contexts/processes.
        for r in rows:r.update(stage='UNKNOWN',scheduler_iteration='UNKNOWN',stage_status='UNVERIFIED_MULTIPLE_NAMESPACES')
    diag=[dict(r) for r in c.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    result=dict(trace=Path(path).name,trace_sha256=file_hash(path),cpu_full_range=full[0],
                gpu_envelope=[min(r['start'] for r in rows),max(r['end'] for r in rows)],
                gpu_identity_status='VERIFIED_METADATA' if identity else 'UNVERIFIED',inventory=inventory,cuda_devices=devices,
                context_ids=sorted(contexts),device_ids=sorted(gpu_ids),global_pids=sorted(pids),
                kernel_count=len(kernels),graph_envelope_count=len(graphs),activity_count=len(rows),
                kernel_time_union_ns=duration([(r['start'],r['end']) for r in kernels]) if not graphs else 'UNAVAILABLE_INCOMPLETE_NODE_TRACE',
                stage_assignment_unknown_count=sum(r['stage']=='UNKNOWN' for r in rows),
                schema={t:sorted(columns(c,t)) for t in tables if t.startswith('CUPTI_ACTIVITY_KIND_') or t in ['NVTX_EVENTS','GPU_METRICS','TARGET_INFO_GPU_METRICS']},
                runtime_api_counts={name:sum(ss.get(r.get('nameId'))==name for r in runtimes) for name in sorted({ss.get(r.get('nameId'),'UNKNOWN') for r in runtimes})},
                diagnostics=diag,clock_policy='All exported timestamps as SessionNs; no guessed offsets.',
                warning='Graph envelopes never substituted for kernel busy time. Correlation match needs Gate-2 completeness audit.')
    save(out/(Path(path).stem+'_timeline.json'),result,overwrite=True)
    save(out/(Path(path).stem+'_activities.json'),rows,overwrite=True)
    fields=['activity_kind','start','end','kernel_name','stage','scheduler_iteration','stage_status','deviceId','contextId','streamId',
            'correlationId','graphId','graphNodeId','gridX','gridY','gridZ','blockX','blockY','blockZ',
            'registersPerThread','staticSharedMemory','dynamicSharedMemory']
    csv_write(out/(Path(path).stem+'_kernel_catalog.csv'),[{k:r.get(k,'UNAVAILABLE') for k in fields} for r in rows],fields)
    stage=[]
    for phase in sorted({r['stage'] for r in rows}):
        rr=[r for r in rows if r['stage']==phase]
        stage.append(dict(stage=phase,kernel_count=sum(r['activity_kind']=='KERNEL' for r in rr),
                          activity_union_ns=duration([(r['start'],r['end']) for r in rr]),
                          scope='stage activity envelope; graphs may contain internal gaps',measurement_status='UNVERIFIED'))
    csv_write(out/(Path(path).stem+'_stage_summary.csv'),stage)
    c.close();return result,rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sqlite');p.add_argument('--output',default=str(ROOT/'analysis'));a=p.parse_args()
    result,_=analyze(a.sqlite,a.output);print(result['trace'],result['kernel_count'],result['stage_assignment_unknown_count'])
