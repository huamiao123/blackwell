"""Apply the residency bounds to verified runtime QKV/probe attributes."""

from pathlib import Path

from resource_model import SPATIAL, V4, load, mixed, read_csv, write_csv
from partition_model import register_partition_fit


def main():
    root = V4 / 'gate4_residency/analysis'
    spatial = SPATIAL / 'qkv_replay_validation'
    probes_path = SPATIAL / 'probe_gpu_validation/raw/probe_20261118_120007_utc/result.json'
    standalone = load(probes_path)
    if standalone['status'] != 'STANDALONE_PROBE_GPU_VERIFIED':
        raise ValueError('probe launch attributes were not verified')
    caps = load(SPATIAL / 'configs/hardware_from_gate3.json')
    granularity = load(root / 'allocation_granularity_audit.json')
    attrs = standalone['device_attributes']
    if (standalone['gpu_uuid'] != caps['uuid'] or attrs['MAX_REGISTERS_PER_MULTIPROCESSOR'] != caps['registers_per_sm'] or
            attrs['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR'] != caps['shared_per_sm']):
        raise ValueError('live device does not match modeled capacities')
    reserved = attrs['RESERVED_SHARED_MEMORY_PER_BLOCK']
    overlaps = {grid: read_csv(spatial / f'analysis/grid{grid}_trace/qkv_pilot_summary.csv') for grid in (32, 128)}
    rows = []
    for name, run_id in [('nvfp4_qkv_proj', 'probe_20261118_141906_utc'), ('bf16_qkv_proj', 'probe_20261118_142102_utc')]:
        result = load(spatial / 'raw' / run_id / 'result.json')
        if result['status'] != 'QKV_REPLAY_OUTPUT_VERIFIED' or result['case'] != name:
            raise ValueError('main function identity could not be established')
        primary = result['live_identity']
        live = primary['register_and_shared_attributes']
        a = dict(threads=primary['block'][0], registers_per_thread=live['NUM_REGS'],
                 static_shared_bytes=live['SHARED_SIZE_BYTES'] + reserved,
                 dynamic_shared_bytes=primary['dynamic_shared_bytes'])
        for probe in standalone['rows']:
            symbol = probe['symbol']
            grid = probe['grid']
            b = dict(threads=probe['threads'], registers_per_thread=probe['attributes']['NUM_REGS'],
                     static_shared_bytes=probe['attributes']['SHARED_SIZE_BYTES'] + reserved,
                     dynamic_shared_bytes=probe['dynamic_shared_bytes'])
            trace = [r for r in overlaps[grid] if r['case'] == name and r['mode'].startswith('parallel')]
            trace_pass = symbol == 'fp32_light' and len(trace) == 2 and all(int(r['overlap_ns']) > 0 for r in trace)
            for n_a in range(1, primary['homogeneous_api_blocks_per_sm'] + 1):
                bounds = mixed(a, b, caps, n_a=n_a, reg_quantum=granularity['register_quantum_per_warp'],
                               shared_quantum=granularity['shared_quantum_bytes'])
                part = register_partition_fit(dict(a, threads=a['threads'] * n_a), b, caps, granularity)
                rows.append(dict(primary=name, probe=symbol, probe_grid=grid, assumed_primary_ctas=n_a,
                                 primary_blocks_per_sm_api=primary['homogeneous_api_blocks_per_sm'],
                                 probe_blocks_per_sm_api=probe['homogeneous_blocks_per_sm'],
                                 driver_reserved_shared_per_cta=reserved,
                                 required_registers=bounds['registers_total'], required_shared_bytes=bounds['shared_total'],
                                 necessary_fit=bounds['same_sm_theoretical_feasibility'],
                                 register_partition_model=part['status'],
                                 register_limit=bounds['registers_limited'], shared_limit=bounds['shared_limited'],
                                 kernel_execution_overlap='VERIFIED' if trace_pass else 'UNVERIFIED_FOR_THIS_PROBE',
                                 same_sm_co_residency='UNVERIFIED_NO_DIRECT_CTA_POSITIONS',
                                 model_scope='live launch attributes, one candidate mixed placement',
                                 overlap_scope='GPU kernel intervals; independent of assumed CTA count',
                                 main_reference=run_id, probe_reference=probes_path.parent.name))
    write_csv(root / 'live_qkv_resource_model.csv', rows)
    print(f'Compared {len(rows)} mixed-resource cases using runtime CTA attributes')


if __name__ == '__main__':
    main()
