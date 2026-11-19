"""Summarize matched E2 request timings without CUDA profiling."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean, median, stdev
import sys

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4 / 'src'))
from common import csv_write, load, percentile


def request_itl(request):
    events = request['output_events']
    if len(events) != 257 or any(row['count'] != 1 for row in events):
        raise ValueError('expected unbundled output per request')
    gaps = [(b['time_ns'] - a['time_ns']) / 1e6 for a, b in zip(events, events[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError('invalid per-request token intervals')
    return gaps


def read_result(root):
    path = root / (root.name + '.json')
    if path.is_file():
        return load(path)
    with gzip.open(path.with_suffix('.json.gz'), 'rt', encoding='utf-8') as source:
        return json.load(source)


def analyze(run, guard):
    if (run['invocation_status'], guard['status'], run['profile']) != ('COMPLETED', 'COMPLETED', 'P0'):
        raise ValueError('completed unprofiled inference required')
    aux = run['gate44']
    condition_order = aux['condition_order']
    if aux['condition'] != 'paired' or len(condition_order) != 20 or len(run['runs']) != 20:
        raise ValueError('expected five complete paired blocks')
    if len(aux['trials']) != len(run['runs']):
        raise ValueError('missing probe trials')

    rows, pairs = [], []
    outputs = None
    reference_graph_count = None
    for trial, probe, condition in zip(run['runs'], aux['trials'], condition_order):
        ordinal = len(rows)
        if (trial['trial'], probe['trial'], probe['condition']) != (ordinal, ordinal, condition):
            raise ValueError('trial order changed')
        expected = len(aux['targets']) if condition in ('parallel', 'serial') else 0
        if (probe['completed_probes'] != expected or
                any(not r['correctness_passed'] for r in probe['completions'] if expected)):
            raise ValueError('incorrect completed work')
        model_work = len(trial['graph_replays']), trial['total_output_tokens']
        if reference_graph_count is None:
            reference_graph_count = model_work
        if model_work != reference_graph_count or model_work[1] != 8 * 257:
            raise ValueError('Graph replay count or request work changed')
        hashes = {r['prompt_hash']: r['output_hash'] for r in trial['request_summary']}
        if outputs is None:
            outputs = hashes
        if len(outputs) != 8 or hashes != outputs:
            raise ValueError('model generated different tokens under the probe')
        summaries = trial['request_summary']
        gaps = [gap for request in trial['requests'].values() for gap in request_itl(request)]
        samples = [r['elapsed_us'] for r in probe['completions']] if expected else []
        cpu_end = trial['full_cpu_end_ns']
        drain_end = trial['post_response_gpu_drain_end_ns']
        if drain_end < cpu_end:
            raise ValueError('invalid drain boundary')
        rows.append(dict(run_id=run['run_id'], precision=run['config']['precision'], phase=aux['phase'],
                         block=ordinal // 4, trial=ordinal, condition=condition,
                         model_wall_ms=trial['full_model_wall_ms'],
                         joint_drain_wall_ms=(drain_end - trial['full_cpu_start_ns']) / 1e6,
                         ttft_mean_ms=mean(r['ttft_ms'] for r in summaries),
                         ttft_p95_ms=percentile([r['ttft_ms'] for r in summaries], .95),
                         tpot_mean_ms=mean(r['tpot_ms'] for r in summaries),
                         itl_p50_ms=percentile(gaps, .5), itl_p95_ms=percentile(gaps, .95),
                         request_p95_ms=percentile([r['model_request_ms'] for r in summaries], .95),
                         completed_probes=expected, probe_flops=expected * aux['probe']['flops'],
                         probe_envelope_mean_us=mean(samples) if samples else 'NOT_APPLICABLE',
                         probe_standalone_mean_us=mean(aux['standalone']['gpu_us']),
                         output_tokens=model_work[1], graph_replays=model_work[0]))
    for start in range(0, len(rows), 4):
        block = {r['condition']: r for r in rows[start:start + 4]}
        if set(block) != {'a_alone', 'empty', 'serial', 'parallel'}:
            raise ValueError('not a complete paired block')
        a, empty, serial, parallel = (block[k] for k in ('a_alone', 'empty', 'serial', 'parallel'))
        if serial['probe_flops'] != parallel['probe_flops']:
            raise ValueError('unequal auxiliary work')
        pairs.append(dict(run_id=run['run_id'], precision=a['precision'], phase=a['phase'], block=a['block'],
                          main_wall_change=parallel['model_wall_ms'] / a['model_wall_ms'] - 1,
                          empty_wall_change=empty['model_wall_ms'] / a['model_wall_ms'] - 1,
                          ttft_change=parallel['ttft_mean_ms'] / a['ttft_mean_ms'] - 1,
                          tpot_change=parallel['tpot_mean_ms'] / a['tpot_mean_ms'] - 1,
                          itl_p95_change=parallel['itl_p95_ms'] / a['itl_p95_ms'] - 1,
                          request_p95_change=parallel['request_p95_ms'] / a['request_p95_ms'] - 1,
                          probe_preflight_to_parallel_event_ratio=(
                              parallel['probe_standalone_mean_us'] / parallel['probe_envelope_mean_us']),
                          serial_parallel_joint_ratio=serial['joint_drain_wall_ms'] / parallel['joint_drain_wall_ms']))
    stats = []
    for metric in list(pairs[0])[4:]:
        values = [row[metric] for row in pairs]
        center = mean(values)
        half = 2.776445 * stdev(values) / 5 ** .5
        stats.append(dict(run_id=run['run_id'], precision=rows[0]['precision'], phase=aux['phase'],
                          metric=metric, n_paired_blocks=5, mean=center, median=median(values),
                          ci95_low=center-half, ci95_high=center+half,
                          scope='five paired blocks using one loaded E2 engine; t interval is descriptive'))
    return rows, pairs, stats


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run_dirs', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = [analyze(read_result(root), load(root / (root.name + '_supervision.json')))
               for root in args.run_dirs]
    for index, name in enumerate(('request_envelopes.csv', 'paired_blocks.csv', 'statistics.csv')):
        csv_write(args.output / name, [row for result in results for row in result[index]])
