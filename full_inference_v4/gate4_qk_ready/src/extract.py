"""Extract actual Q/K intervals from a recorded E2 trace on the CPU."""
import collections
import gzip
import json
from pathlib import Path
import sqlite3
import sys

def extract(db,run,guard):
    db.row_factory=sqlite3.Row
    device,=db.execute('select d.uuid,d.pid,g.uuid as physical from TARGET_INFO_CUDA_DEVICE d join TARGET_INFO_GPU g on g.id=d.gpuId')
    if device['uuid']!=device['physical'] or device['uuid']!=run['gpu_uuid'].removeprefix('GPU-') or device['pid'] not in guard['known_child_pids']:
        raise ValueError('wrong device or worker')
    branches=run['branches']
    names=dict(db.execute('select id,value from StringIds'))
    calls=collections.defaultdict(list)
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        if 'cudaGraphLaunch' in names[r['nameId']]:calls[r['globalTid']].append(r)
    replays={}
    for mark in db.execute("select start,end,globalTid,text from NVTX_EVENTS where text like 'GRAPH_REPLAY|%' and end>start"):
        _,index,step=mark['text'].split('|')
        api,=[v for v in calls[mark['globalTid']] if mark['start']<=v['start'] and v['end']<=mark['end']]
        if api['correlationId'] in replays:raise ValueError('reused correlation ID')
        replays[api['correlationId']]=int(step)
    if sorted(replays.values())!=list(range(1,257)):raise ValueError('incomplete decode')
    runs=collections.defaultdict(list)
    for k in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where graphId>0'):
        if k['correlationId'] in replays:runs[k['correlationId']].append(k)
    reference=None
    results=[]
    launch_resources={}
    for cid,step in replays.items():
        mode=branches['step_modes'][step-1]
        group=runs[cid]
        exe=branches['execs'][mode]
        if len(group)!=branches['kernels'] or any(k['graphId']!=exe or k['graphNodeId']>>32!=exe for k in group):
            raise ValueError('graph exec or node count mismatch')
        nodes={k['graphNodeId']&0xffffffff:k for k in group}
        if len(nodes)!=len(group) or len({(k['globalPid'],k['contextId'],k['deviceId']) for k in group})!=1:
            raise ValueError('invalid per-node or context identity')
        signature={i:tuple(k[f] for f in ('mangledName','gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory','registersPerThread')) for i,k in nodes.items()}
        if reference is not None and signature!=reference:raise ValueError('model kernel signature changed')
        reference=signature
        span=max(k['end'] for k in group)-min(k['start'] for k in group)
        details=[]
        for ref in branches['layers']:
            pred,q,k,cache,att=[nodes[i] for i in ref['nodes']]
            if not results:
                for role,node in [('q',q),('k',k)]:
                    keys=('gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory')
                    launch_resources[f"{ref['layer']}_{role}"]={field:node[field] for field in keys}
            if [names[v['mangledName']] for v in (pred,q,k,cache,att)]!=[ref[x] for x in ('producer','q_name','k_name','kv_consumer','q_consumer')]:
                raise ValueError('bound function changed')
            if min(q['start'],k['start'])<pred['end'] or max(q['end'],k['end'])>min(cache['start'],att['start']):
                raise ValueError('consumer or producer order violated')
            if not any(ref['retained_cache_edge'][0]) and att['start']<cache['end']:
                raise ValueError('ordinary cache-to-attention edge violated')
            if mode!='fork' and q['end']>k['start']:raise ValueError('invalid serial control')
            details.append([ref['layer'],pred['end'],q['start'],q['end'],k['start'],k['end'],cache['start'],att['start'],q['streamId'],k['streamId']])
        results.append(dict(step=step,mode=mode,correlation_id=cid,exec_id=exe,graph_ns=span,layers=details))
    return dict(run=run['run_id'],uuid=device['uuid'],launch_resources=launch_resources,layer_fields=['layer','qkv_end','q_start','q_end','k_start','k_end','cache_start','attention_start','q_stream','k_stream'],time_unit='trace_ns',replays=sorted(results,key=lambda r:r['step']))

if __name__=='__main__':
    rid=sys.argv[1]
    v4=Path.home()/'fp4-llm/results/full_inference_v4'
    run=json.loads((v4/'raw'/(rid+'.json')).read_bytes())
    guard=json.loads((v4/'logs'/(rid+'_supervision.json')).read_bytes())
    if run.get('invocation_status')!='COMPLETED' or guard['status']!='COMPLETED':raise ValueError('unfinished run')
    with sqlite3.connect((v4/'sqlite'/(rid+'.sqlite')).resolve().as_uri()+'?mode=ro',uri=True) as db:
        data=extract(db,run,guard)
    sys.stdout.buffer.write(gzip.compress(json.dumps(data,separators=(',',':')).encode(),mtime=0))
