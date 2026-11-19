"""Freeze layer-18 O and Down from the archived Gate 3 launch evidence."""

from g4_common import SPATIAL, V4, load, read_csv, save, sha


def main():
    inventory_path = V4 / 'gate4_residency/analysis/projection_inventory.csv'
    inventory = read_csv(inventory_path)
    cases = []
    for operator in ('o_proj', 'down_proj'):
        reference_path = SPATIAL / 'qkv_replay_validation' / (operator.removesuffix('_proj') + '_reference.json')
        reference = load(reference_path)
        for precision in ('nvfp4', 'bf16'):
            row, = (r for r in inventory if (r['precision'], r['logical_op'], int(r['layer_id']))
                    == (precision, operator, 18))
            tools_id = int(row['cuda_tools_id'])
            node, = (n for n in reference[precision]['nodes'] if n['tools_id'] == tools_id)
            grid, block = ([int(row[f'{kind}_{axis}']) for axis in 'xyz'] for kind in ('grid', 'block'))
            cta = {key: int(row[key]) for key in ('threads', 'registers_per_thread', 'static_shared_bytes',
                                                 'dynamic_shared_bytes', 'grid_ctas')}
            if (grid, block, cta['dynamic_shared_bytes']) != (node['grid'], node['block'], node['shared_bytes']):
                raise ValueError(f'Gate 3 launch metadata disagreement for {row["module"]}')
            cases.append(dict(case_id=f'{precision}_{operator}', precision=precision, operator=operator,
                              layer=18, module=row['module'], historical_graph_id=int(row['graph_id']),
                              historical_tools_id=tools_id, historical_node_id=int(row['graph_node_id']),
                              grid=grid, block=block, parameter_sha256=row['parameter_sha256'], cta=cta,
                              reference_sha256=sha(reference_path)))
    save(SPATIAL / 'configs/extended_kernel_selection.json',
         dict(schema=1, inventory_sha256=sha(inventory_path), cases=cases))
    print(f'Validated {len(cases)} archived launch and parameter identities for O and Down')


if __name__ == '__main__':
    main()
