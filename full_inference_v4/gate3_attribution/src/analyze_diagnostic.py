"""CPU-only call-scope attribution of one completed Gate-3 Node trace."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
from common import load, save, csv_write, duration, file_hash
from analyze_timeline import analyze, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stem', default='gate3_e2_nvfp4_p2_r02')
    a = p.parse_args()
    stem = a.stem
    output = ROOT / 'analysis'
    raw = load(V4 / 'raw' / (stem + '.json'))
    assert raw['invocation_status'] == 'COMPLETED'
    assert raw['profile'] == 'P2'
    baseline = load(V4 / 'raw/gate2_e2_nvfp4_p2_r01.json')
    for key in ['config_file_sha256', 'prompt_manifest_hash', 'code_sha256', 'gpu_uuid']:
        assert raw[key] == baseline[key], key
    diag = raw['gate3_diagnostic']
    assert diag['wrapper_sha256'] == file_hash(HERE.with_name('diagnostic_worker.py'))
    events = diag['sample_events']
    assert len(events) == 257
    assert Counter(e['phase'] for e in events) == Counter(PREFILL=1, DECODE=256)
    assert all(e['head_apply_calls'] == 1 and e['logits_input_shape'] == [8, 4096] for e in events)
    run = raw['runs'][0]
    assert len(run['requests']) == 8 and len(run['graph_replays']) == 256
    assert all(len(r['output_token_ids']) == 257 for r in run['requests'].values())
    summary, rows = analyze(V4 / 'sqlite' / (stem + '.sqlite'), output)
    assert summary['gpu_identity_status'] == 'VERIFIED_METADATA'
    assert len(summary['global_pids']) == len(summary['context_ids']) == len(summary['device_ids']) == 1
    assert summary['graph_envelope_count'] == 0
    db = sqlite3.connect('file:' + str(V4 / 'sqlite' / (stem + '.sqlite')) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings, nvtx = labels(db)
    samples = [n for n in nvtx if n['label'].startswith('SAMPLE_DISPATCH|')]
    heads = [n for n in nvtx if n['label'].startswith('LOGICAL_OP|LM_HEAD|')]
    assert len(samples) == len(heads) == 257
    apis = defaultdict(list)
    for api in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        apis[api['correlationId']].append(dict(api))
    db.close()

    def containing(ranges, api):
        return [n for n in ranges if n['globalTid'] == api['globalTid']
                and n['start'] <= api['start'] and api['end'] <= n['end']]

    for r in rows:
        r['logical_op'] = 'UNKNOWN'
        r['operator_evidence'] = 'UNVERIFIED'
        r['sample_path'] = False
        candidate = apis.get(r['correlationId'], [])
        if len(candidate) != 1:
            continue
        api = candidate[0]
        scope = containing(samples, api)
        if len(scope) == 1:
            _, iteration, stage = scope[0]['label'].split('|')
            r.update(stage=stage, scheduler_iteration=int(iteration),
                     stage_status='VERIFIED_SAMPLE_CALL_SCOPE_AND_CORRELATION', sample_path=True)
        head = containing(heads, api)
        if len(head) == 1:
            _, _, iteration, stage = head[0]['label'].split('|')
            assert r['sample_path'] and r['scheduler_iteration'] == int(iteration) and r['stage'] == stage
            r.update(logical_op='LM_HEAD', operator_evidence='VERIFIED_ACTUAL_APPLY_CALL_SCOPE_AND_CORRELATION')
    head_rows = [r for r in rows if r['logical_op'] == 'LM_HEAD']
    assert len(head_rows) == 257
    assert all(r['activity_kind'] == 'KERNEL' for r in rows)
    assert not any(r['stage'] == 'UNKNOWN' for r in rows)
    native = [r for r in rows if 'sm120' in r['kernel_name'].lower() and 'e2m1' in r['kernel_name'].lower()]
    assert len(native) == 37008
    with gzip.open(V4 / 'analysis/gate2_e2_nvfp4_p2_r01_activities.json.gz', 'rt') as f:
        previous = json.load(f)
    same_kernels = Counter(r['kernel_name'] for r in rows) == Counter(r['kernel_name'] for r in previous)
    assert same_kernels, 'marker instrumentation changed kernel composition; attribution requires review'
    csv_write(output / 'lm_head_launches.csv', [dict(iteration=r['scheduler_iteration'], stage=r['stage'],
        start_ns=r['start'], end_ns=r['end'], duration_ns=r['end']-r['start'], kernel_name=r['kernel_name'],
        grid_xyz=[r.get(k) for k in ['gridX','gridY','gridZ']], block_xyz=[r.get(k) for k in ['blockX','blockY','blockZ']],
        registers_per_thread=r.get('registersPerThread'), dynamic_shared_bytes=r.get('dynamicSharedMemory'),
        correlation_id=r['correlationId'], operator_evidence=r['operator_evidence']) for r in head_rows])
    categories = {'BACKBONE_AND_PREPARATION': [r for r in rows if not r['sample_path']],
                  'LM_HEAD': head_rows,
                  'OTHER_SAMPLE_OUTPUT_PATH': [r for r in rows if r['sample_path'] and r['logical_op'] != 'LM_HEAD']}
    groups = []
    for stage in ['PREFILL','DECODE']:
        for name, selected in categories.items():
            rr = [r for r in selected if r['stage'] == stage]
            groups.append(dict(stage=stage, category=name, kernel_count=len(rr),
                          kernel_union_ms=duration([(r['start'],r['end']) for r in rr])/1e6,
                          scope='P2 diagnostic interval union; not P0 net time; backbone logical operators unresolved'))
    csv_write(output / 'path_timing.csv', groups)
    with (output / 'attributed_activities.json.gz').open('wb') as f:
        with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0, compresslevel=5) as z:
            z.write(json.dumps(rows, separators=(',', ':')).encode())
    full_range_ms = (summary['cpu_full_range']['end']-summary['cpu_full_range']['start'])/1e6
    head_ms = duration([(r['start'],r['end']) for r in head_rows])/1e6
    result = dict(status='VERIFIED_THIS_DIAGNOSTIC_ONLY', run_id=stem,
        full_model_wall_ms=run['full_model_wall_ms'], previous_p2_model_wall_ms=baseline['runs'][0]['full_model_wall_ms'],
        nvtx_full_window_ms=full_range_ms, total_kernel_count=len(rows),
        native_sm120_e2m1_launches=len(native), graph_replays=len(run['graph_replays']),
        same_kernel_name_multiset_as_gate2=same_kernels,
        old_dispatch_unknown_kernels=summary['stage_assignment_unknown_count'],
        sample_path_kernels=sum(r['sample_path'] for r in rows),
        remaining_stage_unknown=0, lm_head_launches=len(head_rows),
        lm_head_kernel_union_ms=head_ms, lm_head_share_of_p2_full_window=head_ms/full_range_ms,
        runtime_metadata=diag['runtime_metadata'], path_timing=groups,
        limitations=['One E2 NVFP4 diagnostic only, not independent performance repeats',
                    'Backbone QKV/O/FFN per-layer graph node identity remains unverified',
                    'No new GPU Metrics or NCU; no dynamic pipeline/cache/stall counters',
                    'No auxiliary concurrency or useful work validation'])
    save(output / 'diagnostic_validation.json', result, overwrite=True)
    manifest = []
    for path in [V4/'raw'/(stem+'.nsys-rep'), V4/'sqlite'/(stem+'.sqlite')]:
        manifest.append(dict(path=path.relative_to(V4).as_posix(), bytes=path.stat().st_size, sha256=file_hash(path),
                             storage='remote original; not committed'))
    save(output / 'raw_trace_manifest.json', manifest, overwrite=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ['path_timing','limitations','runtime_metadata']}, indent=2))


if __name__ == '__main__':
    main()
