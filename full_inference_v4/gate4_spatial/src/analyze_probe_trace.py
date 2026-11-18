"""Join standalone probe launches to GPU activities without using timing proximity."""

import argparse
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sqlite3
from statistics import median

from g4_common import load, sha, write_csv


LAUNCHES = '''
select coalesce(n.text, label.value) as marker, api.value as api_name,
       name.value as kernel_name, a.start as api_start_ns, a.end as api_end_ns,
       k.start as gpu_start_ns, k.end as gpu_end_ns, k.deviceId, k.contextId,
       k.streamId, k.globalPid, k.correlationId, k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ, k.registersPerThread, k.dynamicSharedMemory
from NVTX_EVENTS n
left join StringIds label on label.id = n.textId
join CUPTI_ACTIVITY_KIND_RUNTIME a
  on a.globalTid = n.globalTid and a.start >= n.start and a.end <= n.end
join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId = a.correlationId
join StringIds api on api.id = a.nameId
join StringIds name on name.id = k.shortName
where coalesce(n.text, label.value) like 'G4_PROBE|%'
order by k.start
'''


def configuration(row):
    return row['symbol'], row['spec']['items'], row['spec']['rounds'], row['grid']


def analyze(trace, run, baseline, output):
    p2 = load(run / 'result.json')
    p0 = load(baseline / 'result.json')
    supervision = load(run / 'supervision.json')
    if (p2['profile'] != 'P2' or p0.get('profile', 'P0') != 'P0' or
            p0['mode'] != p2['mode'] or p2['cubin_sha256'] != p0['cubin_sha256'] or
            p2['gpu_uuid'] != p0['gpu_uuid'] or supervision['status'] != 'COMPLETED' or
            any(r['status'] != 'STANDALONE_PROBE_GPU_VERIFIED' for r in (p0, p2))):
        raise ValueError('invalid paired capture scopes')
    inventory = load(run / 'trace_manifest.json')['files']
    recorded = next(row['sha256'] for row in inventory if row['server_path'].endswith('/calibration.sqlite'))
    if sha(trace) != recorded:
        raise ValueError('SQLite SHA mismatch')
    with closing(sqlite3.connect(trace.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        if [r[0] for r in db.execute('pragma quick_check')] != ['ok']:
            raise ValueError('SQLite integrity check failed')
        physical = db.execute('select id, uuid from TARGET_INFO_GPU').fetchall()
        cuda = db.execute('select gpuId, cudaId, pid, uuid from TARGET_INFO_CUDA_DEVICE').fetchall()
        if len(physical) != 1 or len(cuda) != 1:
            raise ValueError('ambiguous GPU in trace')
        device = cuda[0]
        if (physical[0]['id'] != device['gpuId'] or physical[0]['uuid'] != p2['gpu_uuid'] or
                device['uuid'] != p2['gpu_uuid'] or device['pid'] != supervision['worker_pid']):
            raise ValueError('GPU UUID or supervised PID mismatch')
        activity_count = db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
        launches = [dict(r) for r in db.execute(LAUNCHES)]
    dimensions = {(r['globalPid'], r['contextId'], r['streamId'], r['deviceId']) for r in launches}
    if len(launches) != activity_count or len(dimensions) != 1:
        raise ValueError('missing/extra GPU activity or inconsistent CUDA context')

    event_only = {configuration(r): r for r in p0['rows']}
    profiled = {configuration(r): r for r in p2['rows']}
    if set(event_only) != set(profiled) or len(profiled) != 4:
        raise ValueError('mismatched condition matrix')
    mapped = defaultdict(dict)
    activity = []
    for launch in launches:
        _, symbol, items, rounds, grid, trial = launch['marker'].split('|')
        key = symbol, int(items), int(rounds), int(grid)
        trial = int(trial)
        ref = profiled[key]
        if (trial in mapped[key] or launch['kernel_name'] != symbol or
                launch['api_name'] != 'cuLaunchKernel' or launch['deviceId'] != device['cudaId'] or
                launch['gridX'] != key[-1] or launch['blockX'] != ref['threads'] or
                launch['registersPerThread'] != ref['attributes']['NUM_REGS'] or
                launch['dynamicSharedMemory'] != ref['dynamic_shared_bytes'] or
                tuple(launch[field] for field in ('gridY', 'gridZ', 'blockY', 'blockZ')) != (1, 1, 1, 1) or
                launch['gpu_end_ns'] <= launch['gpu_start_ns']):
            raise ValueError('unexpected probe activity')
        launch['gpu_duration_us'] = (launch['gpu_end_ns'] - launch['gpu_start_ns']) / 1000
        launch['trial'] = trial
        launch['gpu_uuid'] = p2['gpu_uuid']
        mapped[key][trial] = launch
        activity.append(launch)
    if len({r['correlationId'] for r in activity}) != len(activity):
        raise ValueError('nonunique launch correlations')
    summary = []
    for key, ref in sorted(profiled.items()):
        simple = event_only[key]
        if (set(mapped[key]) != set(range(6)) or ref['work_hash'] != simple['work_hash'] or
                ref['output_sha256'] != simple['output_sha256'] or
                not ref['correctness_passed'] or not simple['correctness_passed'] or
                len(ref['elapsed_us']) != 5 or len(simple['elapsed_us']) != 5):
            raise ValueError('work, output, or repeat mismatch')
        times = [mapped[key][trial]['gpu_duration_us'] for trial in range(1, 6)]
        p2_event = ref['elapsed_us']
        delta = median(event - kernel for event, kernel in zip(p2_event, times))
        summary.append(dict(symbol=key[0], items=key[1], rounds=key[2], grid=key[3], n=5,
                            p0_event_median_us=round(median(simple['elapsed_us']), 3),
                            p2_event_median_us=round(median(p2_event), 3),
                            p2_gpu_median_us=round(median(times), 3), p2_gpu_min_us=min(times),
                            p2_gpu_max_us=max(times), p2_event_minus_gpu_median_us=round(delta, 3),
                            work_hash=ref['work_hash'], output_sha256=ref['output_sha256'],
                            trace_sqlite_sha256=recorded))
    write_csv(output / 'qkv_probe_kernel_intervals.csv', activity)
    write_csv(output / 'qkv_probe_trace_summary.csv', summary)
    print(f'Validated {len(activity)} unique GPU launches in {len(summary)} conditions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.trace, args.run, args.baseline, args.output)
