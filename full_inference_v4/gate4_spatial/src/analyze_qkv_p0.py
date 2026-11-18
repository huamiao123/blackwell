"""Aggregate the fixed six-block unprofiled QKV comparison."""

import argparse
from datetime import datetime
from pathlib import Path
import re
from statistics import median

from g4_common import digest, load, sha, write_csv
from metrics import confidence, paired
from plan import MODES


SCOPE = 'P0_GPU_EVENT_ENVELOPE_ISOLATED_QKV'
PRECISIONS = ('nvfp4', 'bf16')
ORDERS = ('ab', 'ba')
METRICS = ('primary_interference', 'probe_throughput_retention', 'joint_speedup')


def telemetry_fields(report):
    out = {}
    for label, key, unit in [('GPU Current Temp', 'temperature_c', 'C'), ('SM', 'sm_clock_mhz', 'MHz'),
                             ('Memory', 'memory_clock_mhz', 'MHz'),
                             ('Instantaneous Power Draw', 'power_w', 'W')]:
        samples = [float(match.group(1)) for row in report['telemetry']
                   if (match := re.search(rf'(?m)^\s*{label}\s*:\s*(\d+(?:\.\d+)?) {unit}\s*$', row['state']))]
        out[key + '_min'] = min(samples) if samples else None
        out[key + '_max'] = max(samples) if samples else None
    return out


def analyze(root, manifest_path, output):
    manifest = load(manifest_path)
    expected = [(block, precision) for block in range(6)
                for precision in (PRECISIONS if block % 2 == 0 else PRECISIONS[::-1])]
    planned = [(row['block'], row['precision']) for row in manifest['planned_order']]
    actual = [(row['block'], row['precision']) for row in manifest['runs']]
    if manifest['status'] != 'COMPLETED' or planned != expected or actual != expected:
        raise ValueError('incomplete or changed predeclared sequence')
    fp32_definition = None
    same_input = None
    sources = None
    blocks = []
    measurements = []
    runs = []
    identifiers = set()
    last_end = None
    for sequence, entry in enumerate(manifest['runs']):
        precision, block, run_id = entry['precision'], entry['block'], entry['run_id']
        run_dir = root / 'raw' / run_id
        result = load(run_dir / 'result.json')
        guard = load(run_dir / 'supervision.json')
        workload = result['overlap_pilot']
        pid = guard['worker_pid']
        begin = datetime.fromisoformat(guard['start'])
        end = datetime.fromisoformat(guard['end'])
        if (pid in identifiers or (last_end and begin < last_end) or end <= begin or entry['launcher_rc'] or
                guard['status'] != 'COMPLETED' or guard['exit_code'] or
                guard['gpu_uuid'] != 'GPU-' + manifest['fixed_gpu'] or guard['telemetry'][0]['compute_pids'] or
                guard['telemetry'][-1]['compute_pids'] or
                any(set(t['compute_pids']) - {pid} for t in guard['telemetry'])):
            raise ValueError('invalid run supervision or process independence')
        identifiers.add(pid)
        last_end = end
        if (workload['profile'] != 'P0' or workload['block'] != block or
                result['case'] != precision + '_qkv_proj' or result['full_model_loaded'] or
                result['status'] != 'QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED' or
                (run_dir / 'trace_manifest.json').exists()):
            raise ValueError('invalid P0 measurement scope')
        definition = workload['fp32_probe']
        code = (guard['worker_sha256'], guard['supervisor_sha256'], workload['overlap_code_sha256'])
        if sequence == 0:
            fp32_definition, same_input, sources = definition, result['input_sha256'], code
        if (definition != fp32_definition or result['input_sha256'] != same_input or code != sources or
                definition['grid'] != 128 or definition['spec']['rounds'] != 8 or definition['flops'] != 16_777_216):
            raise ValueError('source/workload drift across P0 runs')
        work = digest(dict(case=result['case'], input=result['input_sha256'], probe=definition,
                           graph=result['live_identity']['kernel_name'], graph_grid=result['live_identity']['grid'],
                           cache='same layer and buffers in all conditions; no measured L2 state'))
        rows = workload['conditions']
        warmup = workload['warmup']
        if ({row['mode'] for row in rows} != set(MODES) or len(rows) != len(MODES) or
                [row['mode'] for row in warmup] != MODES or not all(row['output_checks_passed'] for row in warmup)):
            raise ValueError('missing or unequal warmup / condition count')
        one = {}
        for condition in rows:
            mode = condition['mode']
            a = int(mode != 'b_alone')
            b = int(mode not in ('a_alone', 'empty_control'))
            if (condition['completed_A'], condition['completed_B']) != (a, b):
                raise ValueError('actual completed work mismatch')
            cooked = dict(profile='P0', measurement_scope=SCOPE, work_hash=work,
                          work_completed=True, correctness_passed=condition['output_checks_passed'],
                          serial_measured=mode.startswith('serial_'), a_completed_units=a, b_completed_units=b,
                          a_ns=condition['a_event_ns'], b_ns=condition['b_event_ns'], joint_ns=condition['joint_event_ns'])
            one[mode] = cooked
            measurements.append(dict(run_id=run_id, precision=precision, block=block, mode=mode, **cooked))
        for order in ORDERS:
            blocks.append(dict(run_id=run_id, precision=precision, block=block, **paired(one, order)))
        runs.append(dict(sequence=sequence, run_id=run_id, precision=precision, block=block, pid=pid,
                         start=guard['start'], end=guard['end'], **telemetry_fields(guard),
                         result_sha256=sha(run_dir / 'result.json')))
    summaries = []
    for order in ORDERS:
        by_precision = {p: {r['block']: r for r in blocks if r['precision'] == p and r['order'] == order}
                        for p in PRECISIONS}
        for field in METRICS:
            for group in (*PRECISIONS, 'nvfp4_minus_bf16'):
                values = ([by_precision[group][i][field] for i in range(6)] if group in PRECISIONS else
                          [by_precision['nvfp4'][i][field] - by_precision['bf16'][i][field] for i in range(6)])
                stats = confidence(values)
                stats['assumption'] = 'six distinct worker/process pairs, t interval assumes independent block effects on this device'
                summaries.append(dict(group=group, order=order, metric=field, **stats,
                                      median=median(values), minimum=min(values), maximum=max(values)))
    write_csv(output / 'qkv_p0_run_inventory.csv', runs)
    write_csv(output / 'qkv_p0_conditions.csv', measurements)
    write_csv(output / 'qkv_p0_block_metrics.csv', blocks)
    write_csv(output / 'qkv_p0_statistics.csv', summaries)
    print(f'Validated {len(runs)} distinct processes, {len(measurements)} conditions, six matched cross-precision blocks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.manifest, args.root / 'analysis')
