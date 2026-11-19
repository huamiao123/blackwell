"""Index Gate 4 evidence under a shared schema, preserving measurement scopes."""

import csv
import gzip
import json
from pathlib import Path
import sys

V4 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V4 / 'src'))
from common import csv_write, load

GATE = V4 / 'gate4'
SPATIAL = V4 / 'gate4_spatial'
REPLAY = SPATIAL / 'qkv_replay_validation'
RESIDENCY = V4 / 'gate4_residency/analysis'
E2 = V4 / 'gate4_inference'
E2_ANALYSIS = E2 / 'analysis/gate44_20261119'
MISSING = 'UNAVAILABLE'
STANDARD = ('run_id', 'timestamp', 'precision', 'primary_operator', 'primary_grid', 'probe_type',
            'probe_grid', 'probe_resource_level', 'threads_per_cta', 'registers_per_thread',
            'dynamic_shared_memory', 'kernel_duration_ms', 'gpu_uuid', 'context_id', 'stream_id',
            'correlation_id', 'overlap_ms', 'A_interference', 'B_throughput_retention',
            'completed_work', 'measurement_scope', 'measurement_tool', 'evidence_status',
            'failure_reason', 'evidence_source')


def records(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def source(path):
    return path.relative_to(V4).as_posix()


def emit(name, rows):
    if not rows:
        raise ValueError(f'missing evidence for {name}')
    columns = {key.casefold(): key for key in STANDARD}
    for row in rows:
        for key in row:
            columns.setdefault(key.casefold(), key)
    filled = []
    for row in rows:
        values = {key.casefold(): value for key, value in row.items()}
        filled.append({field: values.get(key) if values.get(key) not in (None, '') else MISSING
                       for key, field in columns.items()})
    fields = list(columns.values())
    csv_write(GATE / 'analysis' / f'{name}.csv', filled, fields)


def label_probe(filename):
    for symbol in ('tensor_light', 'fp32_heavy'):
        if filename.startswith(symbol):
            return symbol
    return 'fp32_light'


def isolated_runs(primary, uuid):
    conditions, blocks, stats, pressure = [], [], [], []
    for inventory in sorted((REPLAY / 'analysis').glob('*_run_inventory.csv')):
        prefix = inventory.name.removesuffix('_run_inventory.csv')
        path = inventory.with_name(prefix + '_conditions.csv')
        started = {row['run_id']: row['start'] for row in records(inventory)}
        for name, target in [('conditions', conditions), ('block_metrics', blocks), ('statistics', stats)]:
            table = path.with_name(prefix + '_' + name + '.csv')
            for row in records(table):
                precision = row.get('precision', row.get('group'))
                operator = next((op for op in ('gate_up', 'down', 'o') if f'{op}_grid' in prefix), 'qkv') + '_proj'
                launch = primary.get((precision, operator), {})
                b = label_probe(prefix)
                item = dict(row, precision=precision, primary_operator=operator,
                            primary_grid=launch.get('grid_ctas', MISSING), probe_type=b,
                            probe_resource_level=b.split('_')[1], timestamp=started.get(row.get('run_id'), MISSING),
                            gpu_uuid=uuid,
                            A_interference=row.get('primary_interference', MISSING),
                            B_throughput_retention=row.get('probe_throughput_retention', MISSING),
                            completed_work=row.get('b_completed_units', MISSING),
                            measurement_scope=row.get('measurement_scope', 'ISOLATED_PROJECTION_PAIRED_ESTIMATE'),
                            measurement_tool='CUDA Event P0', evidence_status='VERIFIED_SCOPE_ONLY',
                            evidence_source=source(table))
                target.append(item)
                if name == 'block_metrics' and operator == 'qkv_proj' and 'grid' in prefix and b == 'fp32_light':
                    pressure.append(item)
    return conditions, blocks, stats, pressure


def isolated_activity(primary):
    traces = {}
    for path in sorted((REPLAY / 'raw').glob('*/trace_manifest.json')):
        archive = load(path)
        digest = next(row['sha256'] for row in archive['files'] if row['server_path'].endswith('.sqlite'))
        traces[digest] = path.parent
    rows = []
    for summary in sorted((REPLAY / 'analysis').glob('*/qkv_pilot_summary.csv')):
        intervals = {(row['case'], row['mode']): row for row in
                     records(summary.with_name('qkv_pilot_intervals.csv')) if row['role'] == 'B'}
        for item in records(summary):
            b = intervals[item['case'], item['mode']]
            precision, operator = item['case'].split('_', 1)
            run_dir = traces[b['trace_sha256']]
            guard = load(run_dir / 'supervision.json')
            rows.append(dict(item, run_id=run_dir.name, timestamp=guard['start'], precision=precision,
                             primary_operator=operator, primary_grid=primary[precision, operator]['grid_ctas'],
                             probe_type=b['kernel_name'], probe_resource_level=b['kernel_name'].rsplit('_', 1)[-1],
                             gpu_uuid=b['gpu_uuid'], context_id=b['contextId'], stream_id=b['streamId'],
                             correlation_id=b['correlationId'], threads_per_cta=b['blockX'],
                             registers_per_thread=b['registersPerThread'], dynamic_shared_memory=b['dynamicSharedMemory'],
                             kernel_duration_ms=int(b['duration_ns']) / 1e6, overlap_ms=int(item['overlap_ns']) / 1e6,
                             A_interference=item['a_duration_change'], B_throughput_retention=item['b_throughput_retention'],
                             measurement_scope=item['scope'], measurement_tool='P2 CUDA Node Timeline',
                             evidence_status='VERIFIED_TIME_INTERSECTION', evidence_source=source(summary)))
    return rows


def e2_runs(uuid, analysis=E2_ANALYSIS, raw_dir=E2 / 'raw'):
    conditions, blocks, stats, traces, overhead = [], [], [], [], []
    mean_condition = {}
    starts = {}
    for result in sorted(analysis.glob('*/request_envelopes.csv')):
        for name, target in [('request_envelopes.csv', conditions), ('paired_blocks.csv', blocks),
                             ('statistics.csv', stats)]:
            path = result.with_name(name)
            for row in records(path):
                stem = row['run_id']
                if stem not in starts:
                    starts[stem] = load(raw_dir / stem / (stem + '_supervision.json'))['start']
                target.append(dict(row, primary_operator='FULL_E2', probe_type='tensor_light', probe_grid=64,
                                   timestamp=starts[stem], gpu_uuid=uuid,
                                   probe_resource_level='light', A_interference=row.get('main_wall_change', MISSING),
                                   completed_work=row.get('completed_probes', MISSING),
                                   measurement_scope='E2_PRETOKENIZED_' + row['phase'].upper(),
                                   measurement_tool='P0 CPU request and CUDA Event', evidence_status='VERIFIED_SCOPE_ONLY',
                                   evidence_source=source(path)))
        for row in records(result):
            mean_condition.setdefault((row['precision'], row['phase'], row['condition']), []).append(float(row['model_wall_ms']))

    for path in sorted(raw_dir.glob('gate44_*/verified_intersections.json')):
        doc = load(path)
        stem = doc['run_id']
        p = path.parent / (stem + '.json')
        compressed = p.with_suffix('.json.gz')
        if compressed.exists():
            with gzip.open(compressed, 'rt', encoding='utf-8') as stream:
                data = json.load(stream)
            p = compressed
        else:
            data = load(p)
        guard = load(path.parent / (stem + '_supervision.json'))
        for row in doc['rows']:
            traces.append(dict(row, run_id=stem, timestamp=guard['start'], precision=doc['precision'],
                               primary_operator=row['phase'] + '_E2', probe_type='tensor_light', probe_grid=64,
                               probe_resource_level='light', gpu_uuid=doc['device_uuid'], context_id=row['context_id'],
                               stream_id=row['probe_stream_id'], correlation_id=row['probe_correlation'],
                               kernel_duration_ms=(row['probe_end_ns'] - row['probe_start_ns']) / 1e6,
                               overlap_ms=row['overlap_ns'] / 1e6, completed_work=row['probe_tiles'],
                               measurement_scope=doc['phase'], measurement_tool='P2 CUDA Node Timeline',
                               evidence_status='VERIFIED_TIME_INTERSECTION', evidence_source=source(path)))
        key = doc['precision'], doc['phase'], doc['condition']
        base = mean_condition.get(key)
        if not base or ('prefill_parallel_entry' not in data['gate44'] and doc['phase'] != 'decode'):
            continue
        p2 = data['runs'][0]['full_model_wall_ms']
        mean_p0 = sum(base) / len(base)
        overhead.append(dict(run_id=stem, timestamp=guard['start'], precision=key[0], phase=key[1], condition=key[2],
                             p2_wall_ms=p2, mean_p0_wall_ms=mean_p0, p2_over_p0_minus_one=p2 / mean_p0 - 1,
                             measurement_scope='INDEPENDENT_E2_PROFILING_CHECK', measurement_tool='P0 and P2',
                             evidence_status='DESCRIPTIVE_SINGLE_P2', evidence_source=source(p)))
    return conditions, blocks, stats, traces, overhead


def main():
    path = RESIDENCY / 'projection_inventory.csv'
    inventory = records(path)
    uuid = load(SPATIAL / 'configs/hardware_from_gate3.json')['uuid']
    emit('kernel_resource_model', [dict(row, gpu_uuid=uuid, primary_operator=row['logical_op'],
          primary_grid=row['grid_ctas'], threads_per_cta=row['threads'],
          dynamic_shared_memory=row['dynamic_shared_bytes'], measurement_scope='GATE3_ARCHIVED_GRAPH_NODE',
          measurement_tool='CUDA Graph API and P2', evidence_status='VERIFIED_LAUNCH_METADATA',
          evidence_source=source(path)) for row in inventory])
    primary = {(row['precision'], row['logical_op']): row for row in inventory}
    runs, blocks, estimates, pressure = isolated_runs(primary, uuid)
    phases, phase_pairs, phase_estimates, activity, overhead = e2_runs(uuid)
    recheck = E2 / 'analysis/phase_recheck_20261119'
    recheck_raw = E2 / 'raw/phase_recheck_20261119'
    for target, addition in zip((phases, phase_pairs, phase_estimates, activity, overhead),
                                e2_runs(uuid, recheck, recheck_raw)):
        target.extend(addition)
    metrics = []
    for path in sorted((recheck / 'metrics').glob('*.json')):
        doc = load(path)
        stem = doc['run_id']
        guard = load(recheck_raw / stem / (stem + '_supervision.json'))
        for row in doc['rows']:
            metrics.append(dict(row, run_id=stem, precision=doc['precision'], condition=doc['condition'],
                                gpu_uuid=doc['uuid'], timestamp=guard['start'], measurement_scope=row['stage'],
                                measurement_tool='P1 NSYS gb20x 10 kHz', evidence_status='DEVICE_STAGE_SAMPLE',
                                evidence_source=source(path)))
    fees = recheck / 'p1_overhead.csv'
    if fees.exists():
        overhead.extend(dict(row, gpu_uuid=uuid, measurement_tool='P0 and P1',
                             evidence_source=source(fees)) for row in records(fees))
    emit('concurrency_runs', runs + phases)
    emit('paired_blocks', blocks + phase_pairs)
    emit('concurrency_summary', estimates + phase_estimates)
    emit('interference_matrix', [row for row in estimates + phase_estimates
                                if row['metric'] in ('primary_interference', 'probe_throughput_retention',
                                                     'main_wall_change', 'serial_parallel_joint_ratio')])
    emit('probe_pressure_sweep', pressure)
    emit('profiling_overhead', overhead)

    standalone = SPATIAL / 'probe_gpu_validation/analysis/standalone_probe_summary.csv'
    probes = [dict(row, probe_type=row['symbol'], probe_grid=row['grid'], probe_resource_level=row['symbol'].split('_')[1],
                   threads_per_cta=row['threads'], dynamic_shared_memory=row['dynamic_shared_bytes'],
                   completed_work=row['elements_checked_per_launch'], gpu_uuid=uuid,
                   kernel_duration_ms=float(row['median_us']) / 1000,
                   measurement_scope='INDEPENDENT_PROBE', measurement_tool='CUDA Event P0',
                   evidence_status='GPU_CORRECTNESS_VERIFIED', evidence_source=source(standalone))
              for row in records(standalone)]
    dram = SPATIAL / 'analysis/offline_compile/dram_20261119/dram_probe_validation.csv'
    emit('probe_standalone', probes)
    emit('probe_validation', probes + [dict(row, probe_type=row['symbol'], probe_resource_level=row['variant'],
         measurement_scope='OFFLINE_SM120_COMPILE', measurement_tool='nvcc and cuobjdump',
         evidence_status='GPU_UNVERIFIED', evidence_source=source(dram)) for row in records(dram)])
    co_residency = []
    for path in (RESIDENCY / 'live_overlap_evidence_20261119.csv', RESIDENCY / 'live_tensor_extension_20261119.csv'):
        for row in records(path):
            precision, operator = row['primary'].split('_', 1)
            co_residency.append(dict(row, precision=precision, primary_operator=operator, probe_type=row['probe'],
                                     gpu_uuid=uuid, overlap_ms=int(row['observed_overlap_ns']) / 1e6,
                                     measurement_scope='MIXED_CTA_NECESSARY_BOUND_WITH_TIMELINE',
                                     measurement_tool='CUDA Occupancy API and P2', evidence_status=row['same_sm_evidence'],
                                     evidence_source=source(path)))
    emit('co_residency_analysis', co_residency)
    emit('spatial_concurrency', isolated_activity(primary) + activity)
    emit('gpu_metrics', metrics or [dict(gpu_uuid=uuid, measurement_scope='GATE4_DEVICE_METRICS',
                                          evidence_status='UNAVAILABLE', failure_reason='no archived P1 measurements')])
    print('Published 12 scoped Gate 4 CSV tables.')


if __name__ == '__main__':
    main()
