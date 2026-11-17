"""Read all mandatory reports, mappings and implementations; freeze selections."""
from pathlib import Path
from g4_common import *

REQUIRED=[
 'gate2_report.md','gate3_attribution/diagnostic_report.md','gate3_backbone/paired_report.md',
 'gate3_backbone/analysis/graph_operator_mapping.csv','gate3_backbone/analysis/precision_comparison.csv',
 'gate3_backbone/analysis/operator_timing.csv','gate3_backbone/analysis/validation.json',
 'gate3_backbone_bf16/analysis/graph_operator_mapping.csv','gate3_backbone_bf16/analysis/operator_timing.csv',
 'gate3_backbone_bf16/analysis/validation.json','gate3_backbone/src/graph_binding.py',
 'gate3_backbone/src/backbone_worker.py','gate3_backbone/src/analyze_backbone.py',
 'src/run_full_inference.py','src/run_supervised.py','src/collect_nsys.py','src/trace_state.py',
 'gate3_backbone_bf16/src/bf16_observer.py','gate3_backbone_bf16/src/bf16_worker.py',
 'gate3_backbone_bf16/analysis/failed_r01_source.json','analysis/gate2_p0_summary.csv']

def audit():
    records=[]
    for rel in REQUIRED:
        p=V4/rel;text=p.read_text(encoding='utf-8')
        records.append(dict(path=rel,sha256=sha(p),bytes=p.stat().st_size,lines=len(text.splitlines()),read_scope='entire file'))
    for rel in ['audit_v3/source_audit.md','audit_v3/corrected_results.md','audit_v3/raw/gate9_M1024.json']:
        p=V4.parent/rel;p.read_text(encoding='utf-8')
        records.append(dict(path='../'+rel,sha256=sha(p),bytes=p.stat().st_size,read_scope='entire file'))
    cases=[];all_ctas=[]
    comparison=read_csv(V4/'gate3_backbone/analysis/precision_comparison.csv')
    metadata=[]
    for precision,folder in [('nvfp4','gate3_backbone'),('bf16','gate3_backbone_bf16')]:
        rows=read_csv(V4/folder/'analysis/graph_operator_mapping.csv')
        timing=read_csv(V4/folder/'analysis/operator_timing.csv');validation=load(V4/folder/'analysis/validation.json')
        assert len(rows)==144 and len({r['module'] for r in rows})==144
        assert validation['matched_decode_kernel_executions']==36864 and validation['same_kernel_name_multiset_as_gate2']
        metadata.append(validation['gpu_inventory'][0])
        for op in ['qkv_proj','o_proj','gate_up_proj','down_proj']:
            mm=[r for r in rows if r['module'].endswith('.'+op)];assert len(mm)==36
            t=next(r for r in timing if r['logical_op']==op)
            c=next(r for r in comparison if r['precision']==precision and r['logical_op']==op)
            assert all(product(vector(r['grid']))==int(c['grid_ctas']) for r in mm)
            assert all(product(vector(r['block']))==int(c['threads_per_cta']) for r in mm)
            cta=dict(precision=precision,logical_op=op,threads=int(c['threads_per_cta']),
                registers_per_thread=int(c['registers_per_thread']),static_shared_bytes=vector(t['static_shared_bytes'])[0],
                dynamic_shared_bytes=vector(t['dynamic_shared_bytes'])[0],grid_ctas=int(c['grid_ctas']),
                source=folder+'/analysis/operator_timing.csv',source_sha256=sha(V4/folder/'analysis/operator_timing.csv'))
            all_ctas.append(cta)
            if op not in ['qkv_proj','gate_up_proj']:continue
            # Freeze a middle layer before any new measurements; no latency cherry-picking.
            selected=next(r for r in mm if '.layers.18.' in r['module'])
            cases.append(dict(case_id=precision+'_'+op,precision=precision,operator=op,layer=18,module=selected['module'],
                historical_graph_id=int(selected['graph_id']),historical_tools_id=int(selected['tools_id']),
                historical_node_id=int(selected['node_id']),grid=vector(selected['grid']),block=vector(selected['block']),
                parameter_sha256=selected['parameter_sha256'],cta=cta,
                address_scope='historical identity only: never load historical device addresses into a new process',
                timing_scope='new diagnostic single-node replay only after live memory contract; fallback is separately labelled'))
    assert metadata[0]==metadata[1]
    d=metadata[0]
    keymap=dict(sm_count='smCount',registers_per_sm='maxRegistersPerSm',shared_per_sm='maxShmemPerSm',
                warp_size='threadsPerWarp',warps_per_sm='maxWarpsPerSm',ctas_per_sm='maxBlocksPerSm',
                max_threads_per_block='maxThreadsPerBlock',registers_per_block='maxRegistersPerBlock',
                shared_optin_per_block='maxShmemPerBlockOptin',l2_bytes='l2CacheSize',uuid='uuid')
    caps={k:d.get(v,'UNAVAILABLE') for k,v in keymap.items()}
    caps['threads_per_sm']=caps['warp_size']*caps['warps_per_sm']
    caps['threads_per_sm_source']='derived from Gate-3 warp_size × maxWarpsPerSm, not a live device query'
    caps['register_allocation_granularity']='UNAVAILABLE_NOT_QUERIED'
    caps['shared_allocation_granularity']='UNAVAILABLE_NOT_QUERIED'
    caps['source']='gate3_backbone/analysis/validation.json:gpu_inventory[0]'
    caps['source_sha256']=sha(V4/'gate3_backbone/analysis/validation.json')
    save(SPATIAL/'analysis/input_audit.json',dict(files=records,gate4_valid_gpu_evidence_present=False,
        prior_claims='Gate-2/3 only; Gate-4 and useful concurrency not established',gpu_started=False))
    save(SPATIAL/'configs/kernel_selection.json',dict(schema=1,layer_selection_reason='middle layer fixed before new experiments',cases=cases))
    save(SPATIAL/'configs/hardware_from_gate3.json',caps)
    write_csv(V4/'gate4_residency/analysis/cta_resource_table.csv',all_ctas)
    return cases,caps

if __name__=='__main__':
    cases,caps=audit();print('Audited all required artifacts, froze',len(cases),'initial primary kernels; no GPU query')
