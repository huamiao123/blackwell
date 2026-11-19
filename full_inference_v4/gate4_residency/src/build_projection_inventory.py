"""Build the per-layer projection inventory from the frozen Gate 3 traces."""

from ast import literal_eval
from math import ceil, prod
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gate4_spatial/src'))
from g4_common import SPATIAL, V4, load, read_csv, sha, write_csv
from resource_model import homogeneous


def one_observed_value(value):
    values = literal_eval(value)
    if len(values) != 1:
        raise ValueError(f'expected one observed launch configuration, found {values}')
    return values[0]


def inventory():
    caps = load(SPATIAL / 'configs/hardware_from_gate3.json')
    summarized = {
        (row['precision'], row['logical_op']): row
        for row in read_csv(V4 / 'gate4_residency/analysis/cta_resource_table.csv')
    }
    result = []
    for precision, folder in [('nvfp4', 'gate3_backbone'), ('bf16', 'gate3_backbone_bf16')]:
        analysis = V4 / folder / 'analysis'
        timing_path = analysis / 'layer_operator_timing.csv'
        mapping_path = analysis / 'graph_operator_mapping.csv'
        timing_sha, mapping_sha = sha(timing_path), sha(mapping_path)
        mapping = {}
        for row in read_csv(mapping_path):
            parts = row['module'].split('.')
            key = int(parts[2]), parts[-1]
            if parts[:2] != ['model', 'layers'] or key in mapping:
                raise ValueError(f'invalid or duplicate module {row["module"]}')
            mapping[key] = row
        seen = set()
        for timing in read_csv(timing_path):
            layer, op = int(timing['layer_id']), timing['logical_op']
            key = layer, op
            if key in seen:
                raise ValueError(f'duplicate layer timing for {precision} {key}')
            seen.add(key)
            node = mapping[key]
            grid, block = (tuple(one_observed_value(timing[field])) for field in ('grid_xyz', 'block_xyz'))
            if (grid, block) != tuple(tuple(literal_eval(node[field])) for field in ('grid', 'block')):
                raise ValueError(f'Graph Node and timing launch disagree for {precision} {key}')
            if any(len(dims) != 3 or min(dims) <= 0 for dims in (grid, block)):
                raise ValueError(f'invalid launch dimensions for {precision} {key}')
            cta = dict(threads=prod(block), grid_ctas=prod(grid))
            for field in ('registers_per_thread', 'static_shared_bytes', 'dynamic_shared_bytes'):
                cta[field] = int(one_observed_value(timing[field]))
            aggregate = summarized[(precision, op)]
            if any(int(aggregate[field]) != value for field, value in cta.items()):
                raise ValueError(f'layer resource values disagree with the summary for {precision} {key}')
            raw = homogeneous(cta, caps)
            resident_capacity = raw['raw_homogeneous_cta_upper_bound'] * caps['sm_count']
            result.append(dict(
                precision=precision, logical_op=op, layer_id=layer, module=node['module'],
                graph_id=node['graph_id'], graph_node_id=node['node_id'], cuda_tools_id=node['tools_id'],
                parameter_sha256=node['parameter_sha256'], grid_x=grid[0], grid_y=grid[1], grid_z=grid[2],
                block_x=block[0], block_y=block[1], block_z=block[2], **cta,
                warps_per_cta=ceil(cta['threads'] / caps['warp_size']),
                raw_registers_per_cta=cta['threads'] * cta['registers_per_thread'],
                shared_bytes_excluding_runtime_reservation=cta['static_shared_bytes'] + cta['dynamic_shared_bytes'],
                grid_ctas_per_device_sm=cta['grid_ctas'] / caps['sm_count'],
                grid_distinct_sm_fraction_upper_bound=min(cta['grid_ctas'], caps['sm_count']) / caps['sm_count'],
                raw_resident_cta_upper_bound_per_sm=raw['raw_homogeneous_cta_upper_bound'],
                grid_to_optimistic_resident_cta_capacity=cta['grid_ctas'] / resident_capacity,
                optimistic_minimum_grid_waves=ceil(cta['grid_ctas'] / resident_capacity),
                matched_kernel_executions=int(timing['kernel_count']),
                kernel_p50_us=float(timing['kernel_p50_us']), kernel_union_ms=float(timing['kernel_union_ms']),
                actual_sm_distribution='UNAVAILABLE', actual_occupancy='UNAVAILABLE',
                timing_source=timing_path.relative_to(V4).as_posix(), timing_sha256=timing_sha,
                mapping_source=mapping_path.relative_to(V4).as_posix(), mapping_sha256=mapping_sha,
            ))
        validation = load(analysis / 'validation.json')
        if seen != set(mapping) or sum(r['matched_kernel_executions'] for r in result if r['precision'] == precision) != validation['matched_decode_kernel_executions']:
            raise ValueError(f'incomplete Gate 3 attribution for {precision}')
    return result


if __name__ == '__main__':
    rows = inventory()
    write_csv(V4 / 'gate4_residency/analysis/projection_inventory.csv', rows)
    print(f'Validated {len(rows)} layer/operator Graph Node joins across both precisions')
