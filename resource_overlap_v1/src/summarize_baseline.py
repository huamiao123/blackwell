"""Validate fair workload and produce trial/session summaries without GPU access."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    trials, summaries, fairness = [], [], []
    by_batch = collections.defaultdict(dict)
    for file in sorted((root / 'raw').glob('gate1_*.json')):
        d = json.loads(file.read_text())
        if 'error' in d:
            continue
        a, settings = d['args'], d['settings']
        assert not a['profile'] and a['steps'] == 256 and len(d['runs']) >= 5
        assert settings['enable_prefix_caching'] is False and settings['enable_chunked_prefill'] is False
        assert settings['max_num_batched_tokens'] == 16384 and not settings['enforce_eager']
        assert settings['kv_cache_dtype'] == 'fp8_e4m3'
        ids = d['prompt_ids']
        assert len(ids) == a['batch'] and len(set(map(tuple, ids))) == a['batch']
        assert all(len(x) == 2048 for x in ids)
        by_batch[a['batch']][a['precision']] = d
        supervision = json.loads((root / 'logs' / (file.stem + '_supervision.json')).read_text())
        assert supervision['record']['status'] == 'completed' and not supervision['record']['remaining_compute_pids']
        ts = supervision['telemetry']
        group = []
        for run in d['runs']:
            s = run['scheduler']
            decode = [r for r in s if r['decode']]
            nonempty = [r for r in s if r['batch']]
            assert len(nonempty) == 257 and len(decode) == 256
            assert all(r['batch'] == a['batch'] and r['scheduled'] == [1]*a['batch'] for r in decode)
            assert all(all(x >= 2048 for x in r['computed_before']) for r in decode)
            assert nonempty[0]['scheduled'] == [2048]*a['batch']
            assert nonempty[0]['computed_before'] == [0]*a['batch'], 'prefix reuse'
            assert run['output_counts'] == [257]*a['batch'] and run['replays'] == 256
            assert not any(r['preempted'] for r in s)
            # Earlier records predate explicit start/end fields. Reconstruct the
            # documented start from the first exact output interval, not a clock offset.
            begin = run.get('a_start_ns') or round(run['steps'][0]['end_ns'] - run['steps'][0]['latency_ms']*1e6)
            end = run.get('a_end_ns') or round(begin + run['a_wall_ms']*1e6)
            samples = [t for t in ts if begin <= t['monotonic_ns'] <= end]
            power, smclock, temp = [], [], []
            for t in samples:
                text = t['status']
                m = re.search(r'Instantaneous Power Draw\s*:\s*([\d.]+) W', text)
                if not m:
                    m = re.search(r'Average Power Draw\s*:\s*([\d.]+) W', text)
                if m:
                    power.append(float(m[1]))
                # nvidia-smi -q has a current Clocks section followed by SM.
                m = re.search(r'\n\s+Clocks\s*\n(?:(?!\n\s*Applications Clocks).)*?\n\s+SM\s*:\s*([\d.]+) MHz', text, re.S)
                if m:
                    smclock.append(float(m[1]))
                m = re.search(r'GPU Current Temp\s*:\s*([\d.]+) C', text)
                if m:
                    temp.append(float(m[1]))
            row = dict(source=file.name, precision=a['precision'], batch=a['batch'], repeat=run['repeat'],
                       mean_step_ms=run['mean_step_ms'], p50_step_ms=run['p50_step_ms'], p95_step_ms=run['p95_step_ms'],
                       decode_wall_ms=run['a_wall_ms'], tokens_per_s=run['tokens_per_s'], replays=run['replays'],
                       actual_batch=a['batch'], initial_kv=2048, decode_steps=256,
                       prompt_sha256=d['prompt_sha256'], output_sha256=run['output_sha256'],
                       empty_scheduler_calls=len(s)-len(nonempty), telemetry_samples=len(samples),
                       observed_sm_clock_mean_mhz=statistics.mean(smclock) if smclock else None,
                       observed_power_mean_w=statistics.mean(power) if power else None,
                       observed_temperature_mean_c=statistics.mean(temp) if temp else None)
            trials.append(row)
            group.append(row)
        vals = [r['mean_step_ms'] for r in group]
        summaries.append(dict(source=file.name, precision=a['precision'], batch=a['batch'], valid_repeats=len(group),
                              mean_step_ms=statistics.mean(vals), min_step_ms=min(vals), max_step_ms=max(vals),
                              cv_percent=statistics.stdev(vals)/statistics.mean(vals)*100,
                              trial_median_p50_ms=statistics.median(r['p50_step_ms'] for r in group),
                              trial_median_p95_ms=statistics.median(r['p95_step_ms'] for r in group),
                              mean_tokens_per_s=statistics.mean(r['tokens_per_s'] for r in group),
                              mean_decode_total_ms=statistics.mean(r['decode_wall_ms'] for r in group),
                              telemetry_samples=sum(r['telemetry_samples'] for r in group)))
    for batch, ds in by_batch.items():
        assert set(ds) == {'bf16','nvfp4'}
        bf, fp = ds['bf16'], ds['nvfp4']
        assert bf['settings'] == fp['settings']
        assert bf['prompt_ids'] == fp['prompt_ids'] and bf['prompt_sha256'] == fp['prompt_sha256']
        fairness.append(dict(batch=batch, identical_settings=True, identical_token_ids=True,
                             prefix_disabled=True, all_prefill_computed_from_zero=True,
                             actual_batch_verified=True, graph_replays_verified=True,
                             output_hashes_identical=bf['runs'][0]['output_sha256']==fp['runs'][0]['output_sha256']))
    out = root/'analysis'
    for name, rows in [('baseline.csv', trials), ('baseline_summary.csv', summaries)]:
        with (out/name).open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(valid_trials=len(trials), summary=summaries, fairness=fairness,
                measurement_limits=['Five repeated generations per configuration in one process; not five independent model loads.',
                                    'Clock/power observations are sparse two-second supervision samples, not continuous averages.',
                                    'No new CUDA node trace; current kernel composition verification remains pending.',
                                    'Timing boundary differs from historical round2; use these baselines only within this protocol.'])
    (out/'baseline_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
