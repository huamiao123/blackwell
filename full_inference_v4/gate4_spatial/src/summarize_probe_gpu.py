"""Summarize a completed standalone run and refine the historical SMEM bound."""

import argparse
import math
from pathlib import Path
from statistics import median

from g4_common import V4, load, read_csv, sha, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    supervision = load(args.run_dir / 'supervision.json')
    result_path = args.run_dir / 'result.json'
    result = load(result_path)
    if supervision['status'] != 'COMPLETED' or result['status'] != 'STANDALONE_PROBE_GPU_VERIFIED':
        raise ValueError('standalone run was not accepted by its supervisor')
    if supervision['gpu_uuid'] != 'GPU-' + result['gpu_uuid']:
        raise ValueError('device UUID mismatch')

    device = result['device_attributes']
    reserved = int(device['RESERVED_SHARED_MEMORY_PER_BLOCK'])
    summary = []
    rows = result['rows']
    calibration = result.get('mode') == 'qkv_calibration'
    for row in rows:
        if not row['correctness_passed'] or len(row['elapsed_us']) != 5:
            raise ValueError('incomplete standalone validation or timing')
        attrs = row['attributes']
        summary.append(dict(symbol=row['symbol'], grid=row['grid'], threads=row['threads'],
                            registers_per_thread=attrs['NUM_REGS'], function_static_shared_bytes=attrs['SHARED_SIZE_BYTES'],
                            dynamic_shared_bytes=row['dynamic_shared_bytes'], reserved_shared_bytes=reserved,
                            homogeneous_api_blocks_per_sm=row['homogeneous_blocks_per_sm'],
                            elements_checked_per_launch=row['elements_checked'], flops_per_launch=row['flops'],
                            median_us=round(median(row['elapsed_us']), 3), min_us=round(min(row['elapsed_us']), 3),
                            max_us=round(max(row['elapsed_us']), 3), work_hash=row['work_hash'],
                            output_sha256=row['output_sha256'], evidence='STANDALONE_PROBE_ONLY',
                            raw_sha256=sha(result_path)))
        if calibration:
            summary[-1].update(items=row['spec']['items'], rounds=row['spec']['rounds'])
    for work_hash in {r['work_hash'] for r in summary}:
        group = [r for r in summary if r['work_hash'] == work_hash]
        if {r['grid'] for r in group} != {32, 128} or len({r['output_sha256'] for r in group}) != 1:
            raise ValueError('matched grid work/output verification failed')
    if calibration:
        write_csv(args.output / 'qkv_calibration_summary.csv', summary)
        return
    write_csv(args.output / 'standalone_probe_summary.csv', summary)

    granularity = load(V4 / 'gate4_residency/analysis/allocation_granularity_audit.json')
    quantum = granularity['shared_quantum_bytes']
    main_ctas = read_csv(V4 / 'gate4_residency/analysis/cta_resource_table.csv')
    bounds = []
    for cta in main_ctas:
        raw = int(cta['static_shared_bytes']) + int(cta['dynamic_shared_bytes'])
        shared = math.ceil((raw + reserved) / quantum) * quantum
        capacity = device['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR']
        bounds.append(dict(precision=cta['precision'], logical_op=cta['logical_op'], raw_shared_bytes=raw,
                           driver_reserved_bytes=reserved, modeled_allocated_shared_bytes=shared,
                           previous_smem_only_upper_bound=capacity // raw, revised_smem_only_upper_bound=capacity // shared,
                           primary_function_occupancy_api='UNVERIFIED',
                           scope='Gate 3 launch metadata plus live device attribute; shared-memory-only upper bound'))
    write_csv(args.output / 'primary_smem_bounds.csv', bounds)


if __name__ == '__main__':
    main()
