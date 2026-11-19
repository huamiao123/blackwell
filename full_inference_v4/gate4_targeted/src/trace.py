"""Analyze the GPU-side interval of the live layer-18 target and its sibling node."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sqlite3
import sys

def extract(db,run,guard):
    if (run['profile'],run['invocation_status'],guard['status'])!=('P2','COMPLETED','COMPLETED'):
        raise ValueError('incomplete P2')
    gpu,=db.execute('select d.*,g.uuid physical_uuid from TARGET_INFO_CUDA_DEVICE d join TARGET_INFO_GPU g on g.id=d.gpuId')
    if gpu['pid'] not in guard['known_child_pids'] or gpu['uuid']!=gpu['physical_uuid'] or gpu['uuid']!=run['gpu_uuid'].removeprefix('GPU-'):
        raise ValueError('wrong device or process')
    names=dict(db.execute('select id,value from StringIds'))
    calls=defaultdict(list)
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        calls[r['globalTid']].append(r)
    ranges={}
    for mark in db.execute("select * from NVTX_EVENTS where text like 'TARGET_NODE|%' and end>start"):
        _,step,mode=mark['text'].split('|')
        if int(step) in ranges:raise ValueError('duplicate step')
        launches=[c for c in calls[mark['globalTid']] if mark['start']<=c['start'] and c['end']<=mark['end']
                  and 'cudaGraphLaunch' in names[c['nameId']]]
        api,=launches
        ranges[int(step)]=(mode,api['correlationId'])
    ordered=run['target_nodes']['trials'][0]
    if sorted(ranges)!=list(range(33,65)) or [(r['iteration'],r['mode']) for r in ordered]!=[(i,ranges[i][0]) for i in sorted(ranges)]:
        raise ValueError('incomplete replay associations')
    keep={cid for _,cid in ranges.values()};groups=defaultdict(list)
    for k in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where graphId>0'):
        if k['correlationId'] in keep:groups[k['correlationId']].append(k)
    results=[]
    signatures=[]
    for step,(mode,cid) in sorted(ranges.items()):
        meta=run['target_nodes']['variants'][mode]
        ks=groups[cid]
        if not ks or len({(k['globalPid'],k['contextId'],k['deviceId']) for k in ks})!=1 or any(k['graphId']!=meta['exec_id'] for k in ks):
            raise ValueError(f'exec graph mismatch step {step}')
        probe=[k for k in ks if names[k['shortName']]=='tensor_light']
        if len(probe)!=int(mode not in ('sham','original')):raise ValueError('wrong probe count')
        model=[k for k in ks if k not in probe]
        def local_id(k):
            if k['graphNodeId']>>32 != k['graphId']:raise ValueError('unexpected exec node namespace')
            return k['graphNodeId']&0xffffffff
        native=run['target_nodes'].get('native_signature')
        if native:
            observed={str(local_id(k)):[names[k['mangledName']],*[k[f] for f in ('gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory')]] for k in model}
            if observed!=native or len(observed)!=len(model):raise ValueError('per-node native signature changed')
        signatures.append(Counter(tuple(k[f] for f in ('mangledName','gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory')) for k in model))
        by_op={}
        row=dict(step=step,mode=mode,correlation_id=cid,graph_exec_id=meta['exec_id'],model_kernels=len(model),
                 graph_device_span_ns=max(k['end'] for k in ks)-min(k['start'] for k in ks))
        for op,ref in meta['targets'].items():
            a,=[k for k in model if local_id(k)==(ref['tools_id']&0xffffffff)]
            if names[a['mangledName']]!=ref['symbol']:raise ValueError('wrong exact model symbol')
            if [a[f] for f in ('gridX','gridY','gridZ')]!=ref['grid'] or [a[f] for f in ('blockX','blockY','blockZ')]!=ref['block']:
                raise ValueError('wrong bound target launch')
            row[op+'_ns']=a['end']-a['start']
            by_op[op]=a
        if probe:
            op,order=mode.split('_');a=by_op[op];b,=probe
            if (local_id(b),b['gridX'],b['blockX'],b['dynamicSharedMemory'])!=(meta['probe_tools_id']&0xffffffff,64,32,1024):
                raise ValueError('probe identity mismatch')
            start=max(a['start'],b['start']);finish=min(a['end'],b['end'])
            overlap=max(0,finish-start)
            if (order=='ab' and a['end']>b['start']) or (order=='ba' and b['end']>a['start']):
                raise ValueError('serial edge not respected')
            row.update(target_tools_id=a['graphNodeId'],probe_tools_id=b['graphNodeId'],a_start_ns=a['start'],a_end_ns=a['end'],
                       b_start_ns=b['start'],b_end_ns=b['end'],a_stream=a['streamId'],b_stream=b['streamId'],
                       context_id=b['contextId'],b_ns=b['end']-b['start'],overlap_ns=overlap,
                       fraction_b=overlap/(b['end']-b['start']),start_delta_ns=b['start']-a['start'],
                       finish_margin_ns=a['end']-b['end'],fully_inside=b['start']>=a['start'] and b['end']<=a['end'])
        results.append(row)
    if any(s!=signatures[0] for s in signatures):raise ValueError('primary kernel launch signature multiset changed')
    return dict(run_id=run['run_id'],precision=run['config']['precision'],gpu_uuid=gpu['uuid'],rows=results,
                model_kernel_signature_check='identical multiset vs native per target replay',same_sm='UNVERIFIED')

if __name__=='__main__':
    v=Path.home()/'fp4-llm/results/full_inference_v4'
    rid=sys.argv[1]
    with sqlite3.connect((v/'sqlite'/(rid+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        print(json.dumps(extract(db,json.loads((v/'raw'/(rid+'.json')).read_bytes()),json.loads((v/'logs'/(rid+'_supervision.json')).read_bytes()))))
