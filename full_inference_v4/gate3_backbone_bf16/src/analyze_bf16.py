"""CPU-only exact graph-ID join and per-projection diagnostics from full inference."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import statistics
import sys

HERE = Path(__file__).resolve()
ROOT, V4 = HERE.parents[1], HERE.parents[2]
sys.path.insert(0, str(V4/'src'))
from common import load, save, csv_write, duration, file_hash
from analyze_timeline import analyze, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stem', default='gate3_bb_e2_bf16_p2_r01')
    a = p.parse_args(); stem = a.stem; out = ROOT/'analysis'
    raw_path = V4/'raw'/(stem+'.json')
    raw = load(raw_path); base = load(V4/'raw/gate2_e2_bf16_p2_r02.json')
    assert raw['invocation_status'] == 'COMPLETED'
    for key in ['config_file_sha256','prompt_manifest_hash','code_sha256','gpu_uuid']:
        assert raw[key] == base[key], key
    b = raw['backbone_metadata']
    assert b['worker_sha256'] == file_hash(V4/'gate3_backbone/src/backbone_worker.py')
    assert b['binding_source_sha256'] == file_hash(V4/'gate3_backbone/src/graph_binding.py')
    assert raw['bf16_control']['worker_sha256']==file_hash(HERE.with_name('bf16_worker.py'))
    assert raw['bf16_control']['observer_sha256']==file_hash(HERE.with_name('bf16_observer.py'))
    run = raw['runs'][0]
    assert len(run['requests']) == 8 and len(run['graph_replays']) == len(b['replays']) == 256
    assert all(len(r['output_token_ids']) == 257 for r in run['requests'].values())
    used = {r['capture_index'] for r in b['replays']}
    assert len(used) == 1
    capture = b['captures'][next(iter(used))]
    graph = capture['topology']; nodes = capture['matrix_kernels']
    assert len(nodes) == 144
    assert all(n['binding']['status']=='EXACT_UNIQUE_MODULE_REFERENCE' for n in nodes)
    assert all('weight' in {r['tensor'] for r in n['binding']['references']} for n in nodes)
    assert len({n['binding']['module'] for n in nodes}) == 144
    by_tools = {n['tools_id']: n for n in nodes}
    assert Counter(n['binding']['module'].split('.')[-1] for n in nodes) == Counter(qkv_proj=36,o_proj=36,gate_up_proj=36,down_proj=36)
    summary, rows = analyze(V4/'sqlite'/(stem+'.sqlite'), out)
    assert summary['gpu_identity_status'] == 'VERIFIED_METADATA'
    assert len(summary['global_pids']) == len(summary['context_ids']) == len(summary['device_ids']) == 1
    assert summary['graph_envelope_count'] == 0
    db = sqlite3.connect('file:'+str(V4/'sqlite'/(stem+'.sqlite'))+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    _, events = labels(db)
    samples = [e for e in events if e['label'].startswith('SAMPLE_DISPATCH|')]
    heads = [e for e in events if e['label'].startswith('LOGICAL_OP|LM_HEAD|')]
    api = defaultdict(list)
    for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):api[r['correlationId']].append(dict(r))
    db.close()
    def containing(rr, call):
        return [e for e in rr if e['globalTid']==call['globalTid'] and e['start']<=call['start'] and call['end']<=e['end']]
    for r in rows:
        r.update(logical_op='UNKNOWN',layer_id='UNKNOWN',operator_evidence='UNVERIFIED')
        calls = api.get(r['correlationId'],[])
        if len(calls)==1:
            sample = containing(samples,calls[0]); head = containing(heads,calls[0])
            if len(sample)==1:
                _, iteration, phase = sample[0]['label'].split('|')
                r.update(stage=phase,scheduler_iteration=int(iteration))
            if len(head)==1:
                r.update(logical_op='lm_head',operator_evidence='VERIFIED_ACTUAL_HEAD_APPLY_CALL')
        node = by_tools.get(r.get('graphNodeId'))
        if node is not None:
            assert r['graphId']==graph['exec_graph_id'] and r['stage']=='DECODE'
            assert [r['gridX'],r['gridY'],r['gridZ']]==node['grid']
            assert [r['blockX'],r['blockY'],r['blockZ']]==node['block']
            module = node['binding']['module']
            r.update(logical_op=module.split('.')[-1],layer_id=int(module.split('layers.')[1].split('.')[0]),
                     module=module,operator_evidence='VERIFIED_EXACT_WEIGHT_REFERENCE_AND_CUDA_TOOLS_ID')
    projected = [r for r in rows if r['operator_evidence']=='VERIFIED_EXACT_WEIGHT_REFERENCE_AND_CUDA_TOOLS_ID']
    assert len(projected)==144*256
    assert set(Counter(r['module'] for r in projected).values())=={256}
    native = [r for r in rows if 'wmma' in r['kernel_name'].lower() and 'bf16' in r['kernel_name'].lower()]
    assert len(native)==37121 and sum(r['stage']=='UNKNOWN' for r in rows)==0
    with gzip.open(V4/'analysis/gate2_e2_bf16_p2_r02_activities.json.gz','rt') as f:old=json.load(f)
    same_names=Counter(r['kernel_name'] for r in rows)==Counter(r['kernel_name'] for r in old)
    assert same_names
    window=summary['cpu_full_range'];full_ms=(window['end']-window['start'])/1e6
    def summarize(rr,op,layer='ALL'):
        ns=[r['end']-r['start'] for r in rr]
        registers=sorted({r['registersPerThread'] for r in rr})
        return dict(logical_op=op,layer_id=layer,kernel_count=len(rr),kernel_union_ms=duration([(r['start'],r['end']) for r in rr])/1e6,
            kernel_p50_us=statistics.median(ns)/1000,kernel_mean_us=statistics.mean(ns)/1000,
            grid_xyz=sorted({(r['gridX'],r['gridY'],r['gridZ']) for r in rr}),
            block_xyz=sorted({(r['blockX'],r['blockY'],r['blockZ']) for r in rr}),
            registers_per_thread=registers,dynamic_shared_bytes=sorted({r['dynamicSharedMemory'] for r in rr}),
            static_shared_bytes=sorted({r['staticSharedMemory'] for r in rr}),
            scope='one full-inference P2 diagnostic; per-kernel samples are not independent runs')
    operators=[summarize([r for r in projected if r['logical_op']==op],op) for op in ['qkv_proj','o_proj','gate_up_proj','down_proj']]
    layers=[summarize([r for r in projected if r['logical_op']==op and r['layer_id']==layer],op,layer) for layer in range(36) for op in ['qkv_proj','o_proj','gate_up_proj','down_proj']]
    csv_write(out/'operator_timing.csv',operators);csv_write(out/'layer_operator_timing.csv',layers)
    csv_write(out/'graph_operator_mapping.csv',[dict(tools_id=n['tools_id'],graph_id=n['graph_id'],node_id=n['node_id'],module=n['binding']['module'],
        references=n['binding']['references'],grid=n['grid'],block=n['block'],parameter_size=n['parameter_size'],parameter_sha256=n['parameter_sha256']) for n in nodes])
    with (out/'attributed_activities.json.gz').open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=5) as z:z.write(json.dumps(rows,separators=(',',':')).encode())
    with (V4/'raw'/(stem+'.json.gz')).open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=5) as z:z.write(raw_path.read_bytes())
    result=dict(status='VERIFIED_THIS_DECODE_GRAPH_ONLY',run_id=stem,model_wall_ms=run['full_model_wall_ms'],p2_nvtx_window_ms=full_ms,
        replay_capture_index=capture['capture_index'],replay_exec_graph_id=graph['exec_graph_id'],matched_graph_nodes=144,matched_decode_kernel_executions=len(projected),
        unique_modules=144,same_kernel_name_multiset_as_gate2=same_names,total_kernel_count=len(rows),bf16_wmma_family_calls=len(native),
        prefill_projection_calls_without_direct_operator_binding=sum(r['stage']=='PREFILL' and 'cutlass' in r['kernel_name'].lower() and 'bf16' in r['kernel_name'].lower() and r['logical_op']=='UNKNOWN' for r in rows),
        projection_kernel_union_ms=duration([(r['start'],r['end']) for r in projected])/1e6,
        operator_timing=operators,gpu_inventory=summary['inventory'],
        limitation='No NCU/L2/FP32-pipe/actual occupancy or auxiliary concurrency; Prefill per-operator binding not established')
    save(out/'validation.json',result,overwrite=True)
    originals=[dict(path=p.relative_to(V4).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p),storage='remote original, not committed')
        for p in [raw_path,V4/'raw'/(stem+'.nsys-rep'),V4/'sqlite'/(stem+'.sqlite')]]
    save(out/'raw_trace_manifest.json',originals,overwrite=True)
    print(json.dumps({k:v for k,v in result.items() if k!='gpu_inventory'},indent=2))


if __name__=='__main__':main()
