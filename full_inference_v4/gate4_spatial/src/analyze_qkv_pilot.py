"""Validate paired BF16/NVFP4 projection traces, one block per precision."""

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

from g4_common import load, sha, write_csv
from metrics import overlap
from plan import MODES


QUERY = '''
select coalesce(n.text, label.value) as marker, m.value as kernel_name,
       k.start as start_ns, k.end as end_ns, k.deviceId, k.contextId, k.streamId,
       k.globalPid, k.correlationId, k.graphId, k.graphNodeId,
       k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
       k.registersPerThread, k.dynamicSharedMemory
from NVTX_EVENTS n
left join StringIds label on label.id = n.textId
join CUPTI_ACTIVITY_KIND_RUNTIME a
 on a.globalTid = n.globalTid and a.start >= n.start and a.end <= n.end
join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId = a.correlationId
join StringIds m on m.id = k.mangledName
where coalesce(n.text, label.value) like 'G4|%'
order by k.start
'''


def load_run(run, trace):
    result = load(run / 'result.json')
    supervisor = load(run / 'supervision.json')
    manifest = load(run / 'trace_manifest.json')
    expected = next(file['sha256'] for file in manifest['files'] if file['server_path'].endswith('/calibration.sqlite'))
    if sha(trace) != expected or result['status'] != 'QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED':
        raise ValueError('invalid trace or workload')
    if supervisor['status'] != 'COMPLETED' or supervisor['telemetry'][-1]['compute_pids']:
        raise ValueError('supervision did not verify GPU release')
    identity = result['live_identity']
    pilot = result['overlap_pilot']
    probe = pilot['fp32_probe']
    probe_ident = pilot.get('probe_identity', dict(symbol='fp32_light', registers_per_thread=22))
    conditions = {r['mode']: r for r in pilot['conditions']}
    if set(conditions) != set(MODES) or len(pilot['conditions']) != len(MODES):
        raise ValueError('condition matrix mismatch')
    with closing(sqlite3.connect(trace.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if [r[0] for r in db.execute('pragma quick_check')] != ['ok']:
            raise ValueError('invalid SQLite')
        targets = db.execute('select d.*, g.uuid as physical_uuid from TARGET_INFO_CUDA_DEVICE d '
                             'join TARGET_INFO_GPU g on g.id = d.gpuId').fetchall()
        if len(targets) != 1 or targets[0]['pid'] != supervisor['worker_pid']:
            raise ValueError('supervised CUDA PID differs from trace')
        device = targets[0]
        if device['uuid'] != device['physical_uuid'] or 'GPU-' + device['uuid'] != supervisor['gpu_uuid']:
            raise ValueError('contradictory CUDA device mapping')
        launches = [dict(row) for row in db.execute(QUERY)]
    output = {}
    graph_ids = {'A': set(), 'B': set()}
    for row in launches:
        _, case, block, mode, role = row['marker'].split('|')
        dims = [row['gridX'], row['gridY'], row['gridZ']]
        threads = [row['blockX'], row['blockY'], row['blockZ']]
        if (case != result['case'] or block != '0' or mode not in conditions or role not in graph_ids or
                row['deviceId'] != device['cudaId'] or (mode, role) in output or row['end_ns'] <= row['start_ns']):
            raise ValueError('unknown or duplicate pilot activity')
        expected_identity = ((identity['kernel_name'], identity['grid'], identity['block'],
                              identity['register_and_shared_attributes']['NUM_REGS'], identity['dynamic_shared_bytes'])
                             if role == 'A' else (probe_ident['symbol'], [probe['grid'], 1, 1],
                                                  [probe['spec']['threads'], 1, 1], probe_ident['registers_per_thread'],
                                                  probe['spec']['shared_bytes']))
        if (row['kernel_name'], dims, threads, row['registersPerThread'], row['dynamicSharedMemory']) != expected_identity:
            raise ValueError('captured function or launch configuration changed')
        if not row['graphId'] or not row['graphNodeId']:
            raise ValueError('missing Graph identity in pilot trace')
        graph_ids[role].add((row['graphId'], row['graphNodeId']))
        output[(mode, role)] = dict(case=case, role=role, mode=mode, **row, gpu_uuid=device['uuid'],
                                    duration_ns=row['end_ns'] - row['start_ns'], trace_sha256=expected)
    scope = {(r['globalPid'], r['contextId'], r['deviceId']) for r in output.values()}
    if (len(scope) != 1 or any(len(ids) != 1 for ids in graph_ids.values()) or graph_ids['A'] == graph_ids['B'] or
            len({r['correlationId'] for r in output.values()}) != len(output)):
        raise ValueError('invalid isolated graph association')
    summaries = []
    operator = result['case'].split('_', 1)[1].removesuffix('_proj')
    for mode, row in conditions.items():
        a = output.get((mode, 'A'))
        b = output.get((mode, 'B'))
        expected_role_counts = (int(mode != 'b_alone'), int(mode not in ('a_alone', 'empty_control')))
        if ((int(bool(a)), int(bool(b))) != expected_role_counts or
                (row['completed_A'], row['completed_B']) != expected_role_counts or not row['output_checks_passed']):
            raise ValueError('work or output validation missing')
        if not a or not b:
            continue
        intersection, fraction = overlap((a['start_ns'], a['end_ns']), (b['start_ns'], b['end_ns']))
        head = max(0, min(b['end_ns'], a['start_ns']) - b['start_ns'])
        tail = max(0, b['end_ns'] - max(b['start_ns'], a['end_ns']))
        if head + intersection + tail != b['duration_ns']:
            raise ValueError('probe interval decomposition does not sum to elapsed time')
        parallel = mode.startswith('parallel_')
        if (intersection and not parallel) or (parallel and a['streamId'] == b['streamId']):
            raise ValueError('parallel stream or serial negative control does not match protocol')
        a_base = output[('a_alone', 'A')]['duration_ns']
        b_base = output[('b_alone', 'B')]['duration_ns']
        serial = conditions[mode.replace('parallel_', 'serial_')] if parallel else None
        summaries.append(dict(case=result['case'], mode=mode, probe_grid=pilot['fp32_probe']['grid'],
                              a_ns=a['duration_ns'], b_ns=b['duration_ns'],
                              a_alone_ns=a_base, b_alone_ns=b_base, overlap_ns=intersection,
                              fraction_of_shorter=fraction, fraction_of_probe=intersection / b['duration_ns'],
                              b_before_a_ns=head, b_after_a_ns=tail, b_after_a_fraction=tail / b['duration_ns'],
                              a_duration_change=a['duration_ns'] / a_base - 1,
                              b_throughput_retention=b_base / b['duration_ns'], joint_event_ns=row['joint_event_ns'],
                              matched_serial_event_ns=serial['joint_event_ns'] if serial else None,
                              joint_event_speedup=serial['joint_event_ns'] / row['joint_event_ns'] if serial else None,
                              scope=f'ONE_P2_BLOCK_ISOLATED_{operator.upper()}',
                              same_sm='UNVERIFIED'))
    return result, list(output.values()), summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, nargs=2, required=True)
    parser.add_argument('--trace', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = [load_run(run, trace) for run, trace in zip(args.run, args.trace)]
    a, b = (row[0] for row in data)
    if ({a['case'].split('_', 1)[0], b['case'].split('_', 1)[0]} != {'nvfp4', 'bf16'} or
            a['case'].split('_', 1)[1] != b['case'].split('_', 1)[1] or
            a['input_sha256'] != b['input_sha256'] or
            a['overlap_pilot'].get('probe_identity') != b['overlap_pilot'].get('probe_identity') or
            a['overlap_pilot']['fp32_probe'] != b['overlap_pilot']['fp32_probe']):
        raise ValueError('mismatched cross-precision workload')
    write_csv(args.output / 'qkv_pilot_intervals.csv', [r for _, intervals, _ in data for r in intervals])
    write_csv(args.output / 'qkv_pilot_summary.csv', [r for _, _, conditions in data for r in conditions])
    print('Verified, same work and one P2 pilot block per precision')
