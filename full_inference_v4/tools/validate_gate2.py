"""CPU-only publication audit. Never launches CUDA or modifies measured inputs."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    accepted = []
    p0 = Counter()
    requests = 0
    manifest = load(ROOT / 'prompts/manifests/dataset.json')
    manifest_hash = hashlib.sha256(json.dumps(manifest['records'], ensure_ascii=False,
                                              sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert manifest_hash == manifest['manifest_sha256']
    for path in sorted((ROOT / 'raw').glob('gate2_*.json')):
        d = load(path)
        if d.get('schema') != 'v4.1' or d.get('invocation_status') != 'COMPLETED':
            continue
        c = d['config']
        assert c['experiment'] in ('E1', 'E2')
        assert c['engine']['enable_prefix_caching'] is False
        assert c['engine']['enforce_eager'] is False
        assert c['engine']['kv_cache_dtype'] == 'fp8_e4m3'
        assert d['config_file_sha256'] == sha(ROOT / 'configs' / (c['experiment'] + '_' + c['precision'] + '.json'))
        assert d['prompt_manifest_hash'] == manifest_hash
        for filename, expected in d['code_sha256'].items():
            assert sha(ROOT / 'src' / filename) == expected, (path.name, filename)
        for run in d['runs']:
            assert len(run['requests']) == c['batch']
            assert len(run['graph_replays']) == 256
            counts = {rid: [0, 0] for rid in run['requests']}
            for iteration in run['scheduler_iterations']:
                assert not iteration['preempted']
                if iteration['stage'] == 'EMPTY':
                    assert iteration['active_request_count'] == 0
                    assert iteration['scheduled_prefill_tokens'] == iteration['scheduled_decode_tokens'] == 0
                else:
                    assert iteration['active_request_count'] == c['batch']
                for r in iteration['requests']:
                    counts[r['request_id']][0] += r['prefill_tokens']
                    counts[r['request_id']][1] += r['decode_tokens']
            assert all(v == [2048, 256] for v in counts.values())
            assert all(len(q['output_token_ids']) == 257 for q in run['requests'].values())
            if d['profile'] == 'P0':
                requests += len(run['requests'])
        if d['profile'] == 'P0':
            assert len(d['runs']) == 5
            p0[(c['experiment'], c['precision'])] += 1
        accepted.append(d)
    assert len(accepted) == 16 and len(p0) == 4 and set(p0.values()) == {2}
    assert requests == 180
    assert len({json.dumps(d['code_sha256'], sort_keys=True) for d in accepted}) == 1
    assert len({d['gpu_uuid'] for d in accepted}) == 1

    trace_checks = []
    for d in accepted:
        if d['profile'] == 'P0':
            continue
        stem = d['run_id']
        meta = load(ROOT / 'analysis' / (stem + '_timeline.json'))
        v = load(ROOT / 'analysis' / (stem + '_validation.json'))
        assert meta['gpu_identity_status'] == 'VERIFIED_METADATA'
        assert all(len(meta[k]) == 1 for k in ('device_ids', 'context_ids', 'global_pids'))
        with gzip.open(ROOT / 'analysis' / (stem + '_activities.json.gz'), 'rt', encoding='utf-8') as f:
            activity = json.load(f)
        if d['profile'] == 'P2':
            assert len(activity) == meta['kernel_count']
            assert all(a['activity_kind'] == 'KERNEL' for a in activity)
        else:
            assert v['sampling']['gap_count'] == 0
            # SessionNs conversion is not necessarily an integer 100000 ns grid.
            assert abs(v['sampling']['dt_median_ns'] - 100000) <= 100
            assert all(s['timeSourceId'] == 3 for s in v['source_timebase_metadata'])
            with gzip.open(ROOT / 'analysis' / (stem + '_metric_samples.csv.gz'), 'rt', encoding='utf-8') as f:
                samples = sum(1 for _ in csv.DictReader(f))
            assert samples > 100000
        trace_checks.append(dict(run_id=stem, profile=d['profile'], identity=v['gpu_identity'],
                                 kernel_count=v['kernel_count'], sampling=v['sampling']))
    assert len(trace_checks) == 8
    originals = load(ROOT / 'analysis/gate2_raw_trace_manifest.json')
    assert len(originals) == 16 and all(len(r['sha256']) == 64 and r['bytes'] > 0 for r in originals)
    completion = load(ROOT / 'analysis/gate2_completion_state.json')
    assert completion['window_closed'] and not completion['automatic_next_gate']
    assert not completion['permit_files'] and not completion['remaining_own_experiment_processes']
    out = dict(status='PASSED_CPU_PUBLICATION_CHECKS', gpu_launched_by_this_check=False,
               p0_valid_generations=40, p0_requests=180, independent_processes_per_condition=2,
               traces=trace_checks, frozen_source_matches_current=True, prompt_manifest_sha256=manifest_hash,
               original_trace_hashes=16, completion_state=completion,
               limits=['GPU metrics: one capture per condition, no independent metric repeat',
                       'Logical operator/layer assignment incomplete; UNKNOWN activities retained',
                       'No NCU, no auxiliary concurrency, no useful-work acceleration claim',
                       'This publication check does not independently repeat GPU measurements'])
    (ROOT / 'analysis/gate2_quality_validation.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print('PASS: 40 P0 generations, 180 requests, 8 traces; matching sources/configs/prompts; GPU window closed')


if __name__ == '__main__':
    main()
