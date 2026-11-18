"""Estimate the measured time envelope of two real-work candidates from archived P2."""

import csv
import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gate4_spatial/src'))
from g4_common import V4, load, read_csv, sha, write_csv
from metrics import interval_union


def budget(candidate, precision, source, timeline, intervals, evidence):
    window = load(timeline)['cpu_full_range']
    first, last = window['start'], window['end']
    if not intervals or any(not first <= start < end <= last for start, end in intervals):
        raise ValueError('candidate activity outside the original full inference window')
    elapsed = last - first
    active = sum(end - start for start, end in interval_union(intervals))
    return dict(candidate=candidate, precision=precision, gpu_intervals=len(intervals), gpu_union_ns=active,
                same_trace_full_cpu_nvtx_ns=elapsed, apparent_share=active / elapsed,
                ideal_speedup_if_all_identified_time_removed=elapsed / (elapsed - active),
                evidence=evidence, source=source.relative_to(V4).as_posix(), source_sha256=sha(source),
                timeline=timeline.relative_to(V4).as_posix(), timeline_sha256=sha(timeline))


def main():
    analysis = V4 / 'analysis'
    out = []
    for precision, run in [('nvfp4', 'gate2_e2_nvfp4_p2_r01'), ('bf16', 'gate2_e2_bf16_p2_r02')]:
        src = analysis / f'{run}_kernel_catalog.csv.gz'
        with gzip.open(src, 'rt', encoding='utf-8', newline='') as stream:
            rows = csv.DictReader(stream)
            hits = [(int(row['start']), int(row['end'])) for row in rows
                    if row['stage'] == 'PREFILL' and 'embedding' in row['kernel_name'].lower()]
        out.append(budget('prefill_embedding_including_fused_operations', precision, src,
                          analysis / f'{run}_timeline.json', hits, 'P2_KERNEL_NAME_AND_PREFILL_STAGE; NOT EMBEDDING_ONLY'))
    gate3 = V4 / 'gate3_attribution/analysis'
    source = gate3 / 'lm_head_launches.csv'
    rows = read_csv(source)
    if len(rows) != 257 or any(row['operator_evidence'] != 'VERIFIED_ACTUAL_APPLY_CALL_SCOPE_AND_CORRELATION' for row in rows):
        raise ValueError('logits workload identity could not be verified')
    out.append(budget('bf16_lm_head_required_by_nvfp4_inference', 'nvfp4', source,
                      gate3 / 'gate3_e2_nvfp4_p2_r02_timeline.json',
                      [(int(r['start_ns']), int(r['end_ns'])) for r in rows],
                      'P2_ACTUAL_LM_HEAD_APPLY_AND_GPU_CORRELATION'))
    write_csv(V4 / 'gate4_inference/analysis/candidate_trace_budget.csv', out)
    print('Calculated three candidate time envelopes from frozen traces')


if __name__ == '__main__':
    main()
