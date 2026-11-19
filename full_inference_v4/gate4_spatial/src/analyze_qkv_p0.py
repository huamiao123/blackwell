"""Aggregate fixed six-block unprofiled projection comparisons."""

import argparse
from datetime import datetime
from pathlib import Path
import re
from statistics import median

from g4_common import digest, load, sha, write_csv
from metrics import confidence, paired
from plan import MODES
from probe_reference import work_accounting


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
    compare_grids = manifest.get('compare_grids', False)
    low_grid = manifest.get('comparison_grid', 32)
    if compare_grids and low_grid not in [32, 64]:
        raise ValueError('paired grid experiment requires a lower grid')
    operator = manifest.get('operator', 'qkv_proj')
    fixed_grid = manifest.get('fixed_grid', 128)
    probe_symbol = manifest.get('probe_symbol', 'fp32_light')
    if (operator not in ('qkv_proj', 'gate_up_proj') or fixed_grid not in (32, 64, 128)
            or probe_symbol not in ('fp32_light', 'fp32_heavy', 'tensor_light')):
        raise ValueError('unknown primary operator or fixed probe grid')
    scope = SCOPE if operator == 'qkv_proj' else 'P0_GPU_EVENT_ENVELOPE_ISOLATED_GATE_UP'
    grids = [low_grid, 128] if compare_grids else [fixed_grid]
    expected = [(block, precision) for block in range(6)
                for precision in (PRECISIONS if block % 2 == 0 else PRECISIONS[::-1])]
    planned = [(row['block'], row['precision']) for row in manifest['planned_order']]
    actual = [(row['block'], row['precision']) for row in manifest['runs']]
    if manifest['status'] != 'COMPLETED' or planned != expected or actual != expected:
        raise ValueError('incomplete or changed predeclared sequence')
    logical_definition = None
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
        if compare_grids:
            bundle = result['overlap_grid_comparison']
            workloads = bundle['configurations']
            expected_grids = grids if block % 2 == 0 else grids[::-1]
            if (not bundle['same_probe_buffers'] or bundle['order'] != expected_grids or
                    [w['fp32_probe']['grid'] for w in workloads] != expected_grids):
                raise ValueError('grid order or same-buffer requirement failed')
        else:
            workloads = [result['overlap_pilot']]
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
        if (result['case'] != precision + '_' + operator or result['full_model_loaded'] or
                result['status'] != 'QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED' or
                (run_dir / 'trace_manifest.json').exists()):
            raise ValueError('invalid P0 measurement scope')
        for workload in workloads:
            definition = workload['fp32_probe']
            grid = definition['grid']
            if workload['profile'] != 'P0' or workload['block'] != block or grid not in grids:
                raise ValueError('unexpected grid profile or paired block')
            logical = {key: value for key, value in definition.items() if key not in ('grid', 'nonempty_ctas')}
            code = (guard['worker_sha256'], guard['supervisor_sha256'], workload['overlap_code_sha256'])
            if sources is None:
                logical_definition, same_input, sources = logical, result['input_sha256'], code
            identity = workload.get('probe_identity', {'symbol': 'fp32_light'})
            spec = definition['spec']
            if probe_symbol == 'tensor_light':
                fixed_work = (spec.get('probe'), spec.get('tiles'), spec.get('mma_rounds'), definition['flops'])
                expected_work = ('bf16_mma', 256, 128, 268_435_456)
            else:
                fixed_work = (spec.get('probe'), spec.get('items'), spec.get('rounds'), definition['flops'])
                expected_work = ('fp32_fma', 65536, 8, 16_777_216)
            if (logical != logical_definition or result['input_sha256'] != same_input or code != sources or
                    identity['symbol'] != probe_symbol or fixed_work != expected_work or
                    definition['nonempty_ctas'] != grid or
                    any(definition[key] != value for key, value in work_accounting(spec, grid).items())):
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
                cooked = dict(profile='P0', measurement_scope=scope, work_hash=work,
                              work_completed=True, correctness_passed=condition['output_checks_passed'],
                              serial_measured=mode.startswith('serial_'), a_completed_units=a, b_completed_units=b,
                              a_ns=condition['a_event_ns'], b_ns=condition['b_event_ns'], joint_ns=condition['joint_event_ns'])
                one[mode] = cooked
                measurements.append(dict(run_id=run_id, precision=precision, block=block, probe_grid=grid,
                                         mode=mode, **cooked))
            for order in ORDERS:
                blocks.append(dict(run_id=run_id, precision=precision, block=block, probe_grid=grid, **paired(one, order)))
        runs.append(dict(sequence=sequence, run_id=run_id, precision=precision, block=block, pid=pid,
                         start=guard['start'], end=guard['end'], **telemetry_fields(guard),
                         result_sha256=sha(run_dir / 'result.json')))
    summaries = []

    def summarize(group, grid, order, field, values):
        stats = confidence(values)
        stats['assumption'] = 'six matched blocks, t interval assumes independent block effects on this device'
        summaries.append(dict(group=group, probe_grid=grid, order=order, metric=field, **stats,
                              median=median(values), minimum=min(values), maximum=max(values)))

    entries = {(r['precision'], r['probe_grid'], r['order'], r['block']): r for r in blocks}
    for order in ORDERS:
        for field in (*METRICS, 'joint_ns') if compare_grids else METRICS:
            values = {(p, g): [entries[p, g, order, i][field] for i in range(6)] for p in PRECISIONS for g in grids}
            for grid in grids:
                for p in PRECISIONS:
                    summarize(p, grid, order, field, values[p, grid])
                summarize('nvfp4_minus_bf16', grid, order, field,
                          [a - b for a, b in zip(values['nvfp4', grid], values['bf16', grid])])
            if compare_grids:
                for p in PRECISIONS:
                    summarize(p, f'{low_grid}_minus_128', order, field,
                              [a - b for a, b in zip(values[p, low_grid], values[p, 128])])
    if operator != 'qkv_proj':
        prefix = f'gate_up_grid{low_grid}_vs128_p0' if compare_grids else f'gate_up_grid{fixed_grid}_p0'
    elif compare_grids:
        prefix = f'qkv_grid{low_grid}_p0' if low_grid != 32 else 'qkv_grid_p0'
    else:
        prefix = 'qkv_p0' if fixed_grid == 128 else f'qkv_grid{fixed_grid}_fixed_p0'
    if probe_symbol != 'fp32_light':
        prefix = f'{probe_symbol}_{prefix}'
    write_csv(output / f'{prefix}_run_inventory.csv', runs)
    write_csv(output / f'{prefix}_conditions.csv', measurements)
    write_csv(output / f'{prefix}_block_metrics.csv', blocks)
    write_csv(output / f'{prefix}_statistics.csv', summaries)
    print(f'Validated {len(runs)} distinct processes, {len(measurements)} conditions, six matched cross-precision blocks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.manifest, args.root / 'analysis')
