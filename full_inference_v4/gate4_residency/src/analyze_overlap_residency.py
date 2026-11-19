"""Relate validated P2 intervals to necessary mixed-CTA resource bounds."""

import argparse
from math import prod
from pathlib import Path

from partition_model import register_partition_fit
from resource_model import SPATIAL, V4, load, mixed, read_csv, sha, write_csv


def pair_evidence(run_dir, analysis, caps, granularity, reserved):
    result = load(run_dir / 'result.json')
    guard = load(run_dir / 'supervision.json')
    if (result['status'] != 'QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED' or guard['status'] != 'COMPLETED'
            or guard['gpu_uuid'] != 'GPU-' + caps['uuid']):
        raise ValueError('missing validated isolated projection run')
    trace = load(run_dir / 'trace_manifest.json')
    trace_sha = next(r['sha256'] for r in trace['files'] if r['server_path'].endswith('/calibration.sqlite'))
    case = result['case']
    pilot = result['overlap_pilot']
    probe = pilot['probe_identity']
    spec = pilot['fp32_probe']['spec']
    primary = result['live_identity']
    attrs = primary['register_and_shared_attributes']
    a = dict(threads=prod(primary['block']), registers_per_thread=attrs['NUM_REGS'],
             static_shared_bytes=attrs['SHARED_SIZE_BYTES'] + reserved,
             dynamic_shared_bytes=primary['dynamic_shared_bytes'])
    b = dict(threads=spec['threads'], registers_per_thread=probe['registers_per_thread'],
             static_shared_bytes=probe['static_shared_bytes'] + reserved,
             dynamic_shared_bytes=spec['shared_bytes'])
    intervals = {
        (r['mode'], r['role']): r for r in read_csv(analysis / 'qkv_pilot_intervals.csv')
        if r['case'] == case and r['trace_sha256'] == trace_sha
    }
    summary = {r['mode']: r for r in read_csv(analysis / 'qkv_pilot_summary.csv') if r['case'] == case}
    rows = []
    for mode in ('parallel_ab', 'parallel_ba'):
        ai, bi = (intervals[mode, role] for role in ('A', 'B'))
        overlap = max(0, min(int(ai['end_ns']), int(bi['end_ns'])) - max(int(ai['start_ns']), int(bi['start_ns'])))
        if overlap != int(summary[mode]['overlap_ns']):
            raise ValueError('trace summary does not match the selected pair')
        for count in range(1, primary['homogeneous_api_blocks_per_sm'] + 1):
            bounds = mixed(a, b, caps, n_a=count, reg_quantum=granularity['register_quantum_per_warp'],
                           shared_quantum=granularity['shared_quantum_bytes'])
            part = register_partition_fit(dict(a, threads=a['threads'] * count), b, caps, granularity)
            rows.append(dict(run_id=run_dir.name, primary=case, probe=probe['symbol'], mode=mode,
                             primary_grid=prod(primary['grid']), probe_grid=pilot['fp32_probe']['grid'],
                             primary_threads=a['threads'], probe_threads=b['threads'],
                             primary_registers_per_thread=a['registers_per_thread'],
                             probe_registers_per_thread=b['registers_per_thread'],
                             primary_dynamic_shared_bytes=a['dynamic_shared_bytes'],
                             probe_dynamic_shared_bytes=b['dynamic_shared_bytes'],
                             primary_blocks_per_sm_api=primary['homogeneous_api_blocks_per_sm'],
                             probe_blocks_per_sm_api=probe['blocks_per_sm'],
                             assumed_primary_ctas=count, modeled_required_registers=bounds['registers_total'],
                             modeled_required_shared_bytes=bounds['shared_total'],
                             register_limited=bounds['registers_limited'], shared_limited=bounds['shared_limited'],
                             single_primary_cta_shared_exclusion=a['static_shared_bytes'] + a['dynamic_shared_bytes']
                             + b['static_shared_bytes'] + b['dynamic_shared_bytes'] > caps['shared_per_sm'],
                             mixed_resource_bounds=bounds['same_sm_theoretical_feasibility'],
                             register_partition_model=part['status'], observed_overlap_ns=overlap,
                             same_sm_evidence='UNVERIFIED_NO_DIRECT_CTA_POSITIONS',
                             trace_sha256=trace_sha, result_sha256=sha(run_dir / 'result.json')))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', nargs=2, type=Path, action='append', required=True,
                        metavar=('RUN_DIR', 'ANALYSIS_DIR'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    caps = load(SPATIAL / 'configs/hardware_from_gate3.json')
    granularity = load(V4 / 'gate4_residency/analysis/allocation_granularity_audit.json')
    probe_result = load(SPATIAL / 'probe_gpu_validation/raw/probe_20261118_120007_utc/result.json')
    if probe_result['gpu_uuid'] != caps['uuid'] or probe_result['status'] != 'STANDALONE_PROBE_GPU_VERIFIED':
        raise ValueError('live device attributes do not match')
    reserved = probe_result['device_attributes']['RESERVED_SHARED_MEMORY_PER_BLOCK']
    rows = [row for run, analysis in args.pair for row in pair_evidence(run, analysis, caps, granularity, reserved)]
    write_csv(args.output, rows)
    print(f'Joined {len(rows)} modeled placements and observed intervals; CTA positions remain unverified')


if __name__ == '__main__':
    main()
