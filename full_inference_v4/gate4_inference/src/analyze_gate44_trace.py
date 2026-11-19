"""Validate scheduled E2 probe injections against actual CUDA activity."""

import argparse
from collections import defaultdict
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from common import duration, intersection, load, save
from analyze_timeline import labels


def analyze(db, run, guard):
    if run['invocation_status'] != 'COMPLETED' or run['profile'] != 'P2' or guard['status'] != 'COMPLETED':
        raise ValueError('expected completed P2 E2 diagnostic')
    stage = run['gate44']
    trial, = run['runs']
    measurements, = stage['trials']
    completions = {r['iteration']: r for r in measurements['completions']}
    wanted = set(stage['targets'])
    condition = measurements['condition']
    work = stage['probe']
    symbol = work.get('symbol', 'tensor_light')
    if symbol not in ('tensor_light', 'dram_read_light'):
        raise ValueError('unsupported probe')
    shape = (128, 32, 0) if symbol == 'dram_read_light' else (64, 32, 1024)
    device, = db.execute('select d.*,g.uuid as physical_uuid from TARGET_INFO_CUDA_DEVICE d '
                         'join TARGET_INFO_GPU g on g.id=d.gpuId')
    if (device['uuid'] != device['physical_uuid'] or device['pid'] not in guard['known_child_pids']
            or device['uuid'] != run['gpu_uuid'].removeprefix('GPU-')):
        raise ValueError('Nsight process or physical GPU identity mismatch')

    ranges = defaultdict(list)
    _, events = labels(db)
    for label in events:
        name = label['label']
        if name.startswith('G44_B|'):
            _, phase, index, _ = name.split('|')
            index = int(index)
            if index not in completions or completions[index]['phase'] != phase:
                raise ValueError('unexpected probe range')
            ranges[index, 'B'].append(label)
        elif name.startswith('GRAPH_REPLAY|'):
            _, _, index = name.split('|', 2)
            if index.isdecimal() and int(index) in wanted and trial['scheduler_iterations'][int(index)]['stage'] == 'DECODE':
                ranges[int(index), 'A'].append(label)
        elif name.startswith('ITERATION_DISPATCH|'):
            _, index, phase = name.split('|', 2)
            if index.isdecimal() and int(index) in wanted and phase == 'PREFILL':
                ranges[int(index), 'A'].append(label)

    kernels = defaultdict(list)
    for k in db.execute('select k.*,s.value as short_name from CUPTI_ACTIVITY_KIND_KERNEL k '
                        'join StringIds s on k.shortName=s.id order by k.start'):
        kernels[k['correlationId']].append(dict(k))

    def mapped(key):
        window, = ranges[key]
        sql = ('select correlationId from CUPTI_ACTIVITY_KIND_RUNTIME '
               'where globalTid=? and start>=? and end<=?')
        calls = {row[0] for row in db.execute(sql, (window['globalTid'], window['start'], window['end']))}
        return sorted((kernel for call in calls for kernel in kernels[call]), key=lambda row: row['start'])

    rows = []
    for index in sorted(wanted):
        a = mapped((index, 'A'))
        phase = trial['scheduler_iterations'][index]['stage']
        if (not a or any(k['short_name'] == symbol for k in a)
                or (phase == 'DECODE' and (not a[0]['graphId'] or len({k['graphId'] for k in a}) != 1))):
            raise ValueError('unmapped or unexpected primary Graph activities')
        if condition in ('empty', 'a_alone'):
            if condition == 'empty' and mapped((index, 'B')):
                raise ValueError('empty control unexpectedly executed a probe kernel')
            continue
        b, = mapped((index, 'B'))
        if (b['short_name'] != symbol or (b['gridX'], b['blockX'], b['dynamicSharedMemory']) != shape
                or not completions[index]['correctness_passed']):
            raise ValueError('auxiliary launch or output differs from the verified probe')
        if ({k['deviceId'] for k in (*a, b)} != {device['cudaId']}
                or len({(k['globalPid'], k['contextId']) for k in (*a, b)}) != 1
                or b['streamId'] in {k['streamId'] for k in a}):
            raise ValueError('expected the same CUDA context with an independent probe stream')
        active = [(k['start'], k['end']) for k in a]
        probe = (b['start'], b['end'])
        overlap_ns = duration(intersection(active, [probe]))
        if condition == 'serial' and overlap_ns:
            raise ValueError('measured serial control overlaps the target dispatch')
        rows.append(dict(iteration=index, phase=phase, primary_kernels=len(a),
                         primary_graph_id=a[0]['graphId'], primary_stream_ids=sorted({k['streamId'] for k in a}),
                         primary_first_ns=min(k['start'] for k in a), primary_last_ns=max(k['end'] for k in a),
                         primary_union_ns=duration(active), probe_start_ns=probe[0], probe_end_ns=probe[1],
                         probe_stream_id=b['streamId'], context_id=b['contextId'], probe_correlation=b['correlationId'],
                         overlap_ns=overlap_ns, fraction_probe=overlap_ns / (probe[1] - probe[0]),
                         probe_tiles=work.get('tiles',256), same_sm='UNVERIFIED'))
        if symbol == 'dram_read_light':
            rows[-1].update(probe_symbol=symbol, logical_read_bytes=work['logical_read_bytes'])
    expected = 0 if condition in ('a_alone', 'empty') else len(wanted)
    if len(rows) != expected:
        raise ValueError('unexpected number of mapped probe launches')
    return dict(run_id=run['run_id'], precision=run['config']['precision'], phase=stage['phase'],
                condition=condition, device_uuid=device['uuid'], verified_targets=len(wanted),
                probe_activity_count=len(rows), overlap_count=sum(r['overlap_ns'] > 0 for r in rows),
                overlap_total_ns=sum(r['overlap_ns'] for r in rows), rows=rows,
                scope='CUDA launch correlation with GPU kernel interval intersection')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('sqlite', type=Path)
    p.add_argument('run', type=Path)
    p.add_argument('supervision', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    with sqlite3.connect(args.sqlite.resolve().as_uri() + '?mode=ro', uri=True) as database:
        database.row_factory = sqlite3.Row
        report = analyze(database, load(args.run), load(args.supervision))
    save(args.output, report)
    print({key: value for key, value in report.items() if key != 'rows'})
