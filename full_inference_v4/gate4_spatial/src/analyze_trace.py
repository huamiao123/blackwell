"""CPU analysis of NSYS exported SQLite; never attributes by stream length."""
import sqlite3
from contextlib import closing
from collections import defaultdict
from metrics import overlap
from contracts import ContractError

def launch_metadata(kernel,prefix):
    def value(name):
        return kernel[name] if kernel.get(name) is not None else 'UNAVAILABLE'
    def triple(kind):
        fields=[kind+axis for axis in ['X','Y','Z']]
        return [kernel[k] for k in fields] if all(k in kernel and kernel[k] is not None for k in fields) else 'UNAVAILABLE'
    return {prefix+'_grid':triple('grid'),prefix+'_block':triple('block'),
            prefix+'_registers_per_thread':value('registersPerThread'),
            prefix+'_static_shared_bytes':value('staticSharedMemory'),
            prefix+'_dynamic_shared_bytes':value('dynamicSharedMemory')}

def analyze(path,expected_uuid):
    with closing(sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        return _analyze(db,expected_uuid)

def _analyze(db,expected_uuid):
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for t in ['TARGET_INFO_GPU','TARGET_INFO_CUDA_DEVICE','NVTX_EVENTS','StringIds','CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_RUNTIME']:
        if t not in tables:raise ContractError('missing trace table '+t)
    gpu=[dict(r) for r in db.execute('select * from TARGET_INFO_GPU')]
    if len(gpu)!=1 or gpu[0].get('uuid')!=expected_uuid:raise ContractError('GPU identity mismatch')
    cuda_devices=[dict(r) for r in db.execute('select * from TARGET_INFO_CUDA_DEVICE')]
    if len(cuda_devices)!=1 or cuda_devices[0].get('uuid')!=expected_uuid or 'cudaId' not in cuda_devices[0]:
        raise ContractError('unverified CUDA device to physical UUID mapping')
    strings={r['id']:r['value'] for r in db.execute('select * from StringIds')}
    events=[]
    for rr in db.execute('select * from NVTX_EVENTS'):
        r=dict(rr);label=r.get('text') or strings.get(r.get('textId'),'')
        if label.startswith('G4|') and r.get('end') and r['end']>r['start']:r['label']=label;events.append(r)
    calls=defaultdict(list)
    for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):calls[r['correlationId']].append(dict(r))
    kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
    if any(k.get('deviceId')!=cuda_devices[0]['cudaId'] for k in kernels):raise ContractError('kernel CUDA ordinal mismatch')
    if len({(k.get('globalPid'),k.get('contextId'),k.get('deviceId')) for k in kernels})!=1:
        raise ContractError('multiple process/context/device namespaces')
    mapped=defaultdict(list);unmapped=[]
    for k in kernels:
        scopes={e['label'] for c in calls.get(k['correlationId'],[]) for e in events
                if c['globalTid']==e['globalTid'] and e['start']<=c['start'] and c['end']<=e['end']}
        if len(scopes)!=1:unmapped.append(k);continue
        label=next(iter(scopes));parts=label.split('|')
        if len(parts)!=5 or parts[4] not in ['A','B']:raise ContractError('invalid G4 trace label')
        k['kernel_name']=strings.get(k.get('demangledName'),'UNAVAILABLE')
        mapped[tuple(parts[1:4])+(parts[4],)].append(k)
    result=[]
    for case,block,mode in sorted({key[:3] for key in mapped}):
        aa=mapped.get((case,block,mode,'A'),[]);bb=mapped.get((case,block,mode,'B'),[])
        if len(aa)!=1 or len(bb)!=1:
            if mode.startswith('parallel'):
                raise ContractError('not a single-kernel pair; use separately labelled stage/Graph analysis')
            continue  # alone/empty controls do not define a two-kernel overlap
        a,b=aa[0],bb[0]
        ns,ratio=overlap((a['start'],a['end']),(b['start'],b['end']))
        if ns and (mode.startswith('serial') or mode in ['same_stream_control','non_overlap_control']):
            raise ContractError('negative control has overlap; experiment must be reviewed')
        if a['contextId']!=b['contextId']:raise ContractError('cross-context comparison')
        if mode.startswith('parallel') and a['streamId']==b['streamId']:raise ContractError('parallel streams are identical')
        result.append(dict(case_id=case,block=block,mode=mode,overlap_ns=ns,overlap_ratio=ratio,
            a_start_ns=a['start'],a_end_ns=a['end'],b_start_ns=b['start'],b_end_ns=b['end'],
            a_stream=a['streamId'],b_stream=b['streamId'],context=a['contextId'],device_uuid=expected_uuid,
            a_kernel=a['kernel_name'],b_kernel=b['kernel_name'],a_duration_ns=a['end']-a['start'],b_duration_ns=b['end']-b['start'],
            **launch_metadata(a,'a'),**launch_metadata(b,'b'),
            evidence_status='CONCURRENT_KERNEL_EXECUTION_VERIFIED' if ns else 'NO_INTERVAL_OVERLAP_IN_THIS_TRACE',
            primary_identity_status='REQUIRES_SEPARATE_LIVE_NODE_PROOF_NOT_INFERRED_FROM_NAME',
            same_sm='SAME_SM_CO_RESIDENCY_UNVERIFIED'))
    return result,unmapped
