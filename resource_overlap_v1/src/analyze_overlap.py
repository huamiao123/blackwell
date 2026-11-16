"""CPU analysis of per-trial raw results and actual CUPTI interval overlap."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics


def union(intervals):
    result = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if result and s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], e)
        else:
            result.append([s, e])
    return result


def intersection(a, b):
    a, b = union(a), union(b)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            result.append([s, e])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(xs):
    return sum(e - s for s, e in union(xs))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def timeline(path, out):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    c.row_factory = sqlite3.Row
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    schema = {t: [dict(r) for r in c.execute(f'PRAGMA table_info("{t}")')]
              for t in ['NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL', 'TARGET_INFO_GPU', 'TARGET_INFO_CUDA_DEVICE']}
    assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables, 'No node-level kernel evidence'
    strings = {r['id']: r['value'] for r in c.execute('SELECT * FROM StringIds')}
    nvtx = []
    for rr in c.execute('SELECT * FROM NVTX_EVENTS'):
        r = dict(rr)
        r['label'] = r.get('text') or strings.get(r.get('textId'), '')
        nvtx.append(r)
    aw = [r for r in nvtx if r['label'].startswith('OVERLAP_A|') and r.get('end', 0) > r['start']]
    jw = [r for r in nvtx if r['label'].startswith('OVERLAP_JOINT|') and r.get('end', 0) > r['start']]
    assert len(aw) == len(jw) == 1, ('Expected one profile trial', len(aw), len(jw))
    start, end = aw[0]['start'], aw[0]['end']
    kernels = []
    for rr in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>? AND start<?', (jw[0]['start'], jw[0]['end'])):
        r = dict(rr)
        r['kernel_name'] = strings.get(r.get('demangledName'), strings.get(r.get('shortName'), ''))
        r['task'] = 'B' if re.search(r'_aux_(fp32|dram|tensor|empty)', r['kernel_name']) else 'A'
        kernels.append(r)
    assert kernels, 'Empty trace'
    inventories = [dict(r) for r in c.execute('SELECT * FROM TARGET_INFO_GPU')]
    devices = [dict(r) for r in c.execute('SELECT * FROM TARGET_INFO_CUDA_DEVICE')]
    identity = len(inventories) == 1 and inventories[0].get('busLocation') == '0000:27:00.0' and all(
        d['uuid'] == inventories[0]['uuid'] for d in devices)
    assert identity, 'GPU identity not established'
    device_ids = {d['cudaId'] for d in devices} if devices and 'cudaId' in devices[0] else None
    aa = [(max(start, r['start']), min(end, r['end'])) for r in kernels if r['task'] == 'A']
    bb = [(max(start, r['start']), min(end, r['end'])) for r in kernels if r['task'] == 'B']
    aa, bb = union(aa), union(bb)
    both = intersection(aa, bb)
    bnodes = [r for r in kernels if r['task'] == 'B']
    anodes = [r for r in kernels if r['task'] == 'A']
    summary = collections.Counter(r['kernel_name'] for r in anodes)
    diag = [dict(r) for r in c.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    severe = [r for r in diag if r.get('severity', 0) >= 3 or re.search(r'overflow|missing data|data loss', r.get('text', ''), re.I)]
    result = dict(file=path.name, schema=schema, a_window=aw[0], joint_window=jw[0],
                  gpu_identity_verified=identity, gpu_inventory=inventories, cuda_devices=devices,
                  kernel_device_ids=sorted({r['deviceId'] for r in kernels}),
                  a_streams=sorted({r['streamId'] for r in anodes}), b_streams=sorted({r['streamId'] for r in bnodes}),
                  a_kernel_count=len(anodes), b_kernel_count=len(bnodes),
                  a_kernel_union_ns=duration(aa), b_kernel_union_inside_a_ns=duration(bb),
                  physical_intersection_ns=duration(both),
                  intersection_fraction_of_b_inside_a=duration(both)/duration(bb) if duration(bb) else None,
                  kernel_coverage=duration(aa)/(end-start),
                  b_fully_contained_nodes=sum(start <= r['start'] and r['end'] <= end for r in bnodes),
                  b_partial_nodes=sum(r['start'] < end and r['end'] > start and not (start <= r['start'] and r['end'] <= end) for r in bnodes),
                  gpu_first_start_skew_ns=min(r['start'] for r in bnodes)-min(r['start'] for r in anodes) if bnodes and anodes else None,
                  kernel_families=summary.most_common(), diagnostics=diag, severe_diagnostics=severe,
                  raw_interval_timebase='NSYS SessionNs start/end; no offsets',
                  same_sm_residency='not established by this trace',
                  overlap_verified=bool(both) and not severe,
                  interpretation='Diagnostic profiled run only; not formal performance')
    # Preserve compact node evidence and all launch-resource fields in CSV.
    fields = ['start', 'end', 'task', 'kernel_name', 'deviceId', 'streamId', 'gridX', 'gridY', 'gridZ',
              'blockX', 'blockY', 'blockZ', 'registersPerThread', 'staticSharedMemory', 'dynamicSharedMemory', 'correlationId']
    write_csv(out / (path.stem + '_kernels.csv'), [{k:r.get(k) for k in fields} for r in kernels])
    (out / (path.stem + '_timeline.json')).write_text(json.dumps(result, indent=2))
    return result


def analyze(root):
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    baseline, standalone, paired, all_rows = [], [], [], []
    for file in sorted((root / 'raw').glob('*.json')):
        d = json.loads(file.read_text())
        if 'error' in d:
            continue  # Failed invocations remain in raw; never become samples.
        if 'precision' not in d.get('args', {}):
            if 'description' in d:
                for r in d.get('runs', []):
                    standalone.append(dict(source=file.name, kind=d['args']['kind'], repeat=r['repeat'],
                                           event_ms=r['event_ms'], wall_ms=r['wall_ms'],
                                           config=json.dumps(d['description']['config'], sort_keys=True),
                                           work=json.dumps(d['description']['work'], sort_keys=True)))
            continue
        profile = d['args'].get('profile', False)
        for r in d['runs']:
            flat = dict(source=file.name, precision=d['args']['precision'], batch=d['args']['batch'],
                        decode_steps=d['args']['steps'], profile=profile, probe=d['args']['probe'],
                        repeat=r['repeat'], mode=r['mode'], prompt_sha256=d['prompt_sha256'])
            flat.update({k:v for k,v in r.items() if k in ['a_wall_ms', 'mean_step_ms', 'p50_step_ms', 'p95_step_ms',
                                                        'tokens_per_s', 'joint_wall_ms', 'b_span_ms', 'replays', 'b_done_at_a_end']})
            all_rows.append(flat)
            if r['mode'] == 'alone_a' and not profile:
                baseline.append(flat)
        if profile:
            continue
        groups = collections.defaultdict(dict)
        for r in d['runs']:
            groups[r['repeat']][r['mode']] = r
        for rep, rows in groups.items():
            if 'alone_a' not in rows or 'alone_b' not in rows:
                continue
            aa, bb = rows['alone_a'], rows['alone_b']
            serial = {mode: rows[mode]['joint_wall_ms'] for mode in ['serial_ab', 'serial_ba'] if mode in rows}
            for mode in ['parallel_ab', 'parallel_ba', 'empty_ab', 'empty_ba']:
                if mode not in rows:
                    continue
                r = rows[mode]
                paired.append(dict(source=file.name, precision=d['args']['precision'], batch=d['args']['batch'],
                                   decode_steps=d['args']['steps'], probe=d['args']['probe'], repeat=rep, mode=mode,
                                   ia=r['a_wall_ms']/aa['a_wall_ms']-1,
                                   rb=bb['b_span_ms']/r['b_span_ms'] if not mode.startswith('empty') else None,
                                   s_fastest_serial=min(serial.values())/r['joint_wall_ms'] if serial and not mode.startswith('empty') else None,
                                   s_serial_ab=serial.get('serial_ab', 0)/r['joint_wall_ms'] if 'serial_ab' in serial and not mode.startswith('empty') else None,
                                   s_serial_ba=serial.get('serial_ba', 0)/r['joint_wall_ms'] if 'serial_ba' in serial and not mode.startswith('empty') else None,
                                   p95_ratio=r['p95_step_ms']/aa['p95_step_ms'],
                                   b_done_at_a_end=r['b_done_at_a_end'],
                                   same_finite_b_work=not mode.startswith('empty')))
    write_csv(out / 'baseline.csv', baseline)
    write_csv(out / 'auxiliary_standalone.csv', standalone)
    write_csv(out / 'overlap_results.csv', all_rows)
    write_csv(out / 'interference_matrix.csv', paired)
    traces = [timeline(p, out) for p in sorted((root / 'sqlite').glob('*.sqlite'))]
    (out / 'timeline_validation.json').write_text(json.dumps(traces, indent=2))
    print(json.dumps(dict(baselines=len(baseline), standalone=len(standalone), paired=len(paired),
                          traces=[dict(file=t['file'], overlap=t['overlap_verified'],
                                       intersection_ms=t['physical_intersection_ns']/1e6) for t in traces]), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    analyze(a.root)
