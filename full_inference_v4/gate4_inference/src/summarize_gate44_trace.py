"""Join verified Gate 4.4 activity with equal-work serial GPU controls."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from common import csv_write, load


def summarize(raw, excluded=()):
    groups = defaultdict(dict)
    for path in sorted(raw.glob('gate44_*/verified_intersections.json')):
        row = load(path)
        if row['run_id'] in excluded:
            continue
        key = row['precision'], row['phase']
        condition = row['condition']
        if condition in groups[key]:
            raise ValueError('duplicate verified case; choose a unique raw set')
        manifest = load(path.parent / (row['run_id'] + '.sqlite.manifest.json'))
        groups[key][condition] = (row, manifest['sha256'])
    result = []
    for (precision, phase), runs in sorted(groups.items()):
        if 'parallel' not in runs:
            continue
        parallel, parallel_sha = runs['parallel']
        serial, serial_sha = runs.get('serial', (None, 'UNAVAILABLE'))
        p = {row['iteration']: row for row in parallel['rows']}
        s = {row['iteration']: row for row in serial['rows']} if serial else {}
        if serial and (set(p) != set(s) or serial['overlap_count']):
            raise ValueError('incorrect serial comparison')
        for stage in sorted({row['phase'] for row in p.values()}):
            indexes = [i for i, row in p.items() if row['phase'] == stage]
            if serial and any(p[i]['primary_kernels'] != s[i]['primary_kernels'] for i in indexes):
                raise ValueError('primary activity count differs across P2 controls')
            p_us = [(p[i]['probe_end_ns'] - p[i]['probe_start_ns']) / 1000 for i in indexes]
            s_us = [(s[i]['probe_end_ns'] - s[i]['probe_start_ns']) / 1000 for i in indexes] if serial else []
            result.append(dict(precision=precision, phase_condition=phase, actual_stage=stage,
                               parallel_run=parallel['run_id'], parallel_sqlite_sha256=parallel_sha,
                               serial_run=serial['run_id'] if serial else 'UNAVAILABLE', serial_sqlite_sha256=serial_sha,
                               gpu_verified_probe_calls=len(indexes),
                               calls_with_overlap=sum(p[i]['overlap_ns'] > 0 for i in indexes),
                               overlap_total_us=sum(p[i]['overlap_ns'] for i in indexes) / 1000,
                               mean_probe_overlap_fraction=mean(p[i]['fraction_probe'] for i in indexes),
                               parallel_probe_kernel_median_us=median(p_us),
                               serial_probe_kernel_median_us=median(s_us) if s_us else 'UNAVAILABLE',
                               probe_serial_context_rate_ratio=sum(s_us) / sum(p_us) if s_us else 'UNAVAILABLE',
                               parallel_primary_gpu_span_ms=sum(p[i]['primary_last_ns'] - p[i]['primary_first_ns']
                                                                for i in indexes) / 1e6,
                               parallel_primary_kernel_union_ms=sum(p[i]['primary_union_ns'] for i in indexes) / 1e6,
                               serial_primary_gpu_span_ms=(sum(s[i]['primary_last_ns'] - s[i]['primary_first_ns']
                                                               for i in indexes) / 1e6 if s_us else 'UNAVAILABLE'),
                               serial_primary_kernel_union_ms=(sum(s[i]['primary_union_ns'] for i in indexes) / 1e6
                                                                if s_us else 'UNAVAILABLE'),
                               scope='one P2 run per condition; actual same-work probe kernel durations'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('raw', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-run', action='append', default=[])
    args = p.parse_args()
    rows = summarize(args.raw, args.exclude_run)
    if not rows:
        raise ValueError('no verified parallel activity')
    csv_write(args.output, rows)
