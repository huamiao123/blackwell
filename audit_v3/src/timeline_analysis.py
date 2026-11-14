"""Physical overlap from a GPU timeline (torch.profiler chrome trace).

Two different quantities, never conflated:

  task_overlap_ms         intersection of the two tasks' overall [start, end] spans.
                          Says "they were both running in the same window", NOT
                          that their kernels ran at the same instant.

  kernel_overlap_ms       time during which at least one kernel of A and at least
                          one kernel of B were executing simultaneously.
                          This is the only defensible evidence of concurrent
                          kernel execution. It still does not prove co-residency
                          on the same SM.

Also reported: per-stream busy time, gaps, and the tail after one side finished.

Caveat: the profiler itself perturbs execution (CUPTI). Treat these numbers as a
cross-check on the event-based timings, not as the primary measurement.
"""
import json


def _union(intervals):
    iv = sorted(intervals)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _length(iv):
    return sum(e - s for s, e in iv)


def _intersection(a, b):
    a, b = _union(a), _union(b)
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def load_kernels(trace_path):
    ev = json.load(open(trace_path))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel" and "ts" in e and "dur" in e]
    by_stream = {}
    for e in ks:
        st = e["args"].get("stream")
        by_stream.setdefault(st, []).append((e["ts"], e["ts"] + e["dur"], e.get("name", "")))
    return by_stream


def streams_by_kernel_pattern(trace_path, pattern_A, pattern_B, flags=0):
    """Identify streams by matching kernel names, instead of guessing from counts.

    Returns (stream_A, stream_B) or (None, None) when the match is ambiguous.
    """
    import re
    by = load_kernels(trace_path)
    hitA = [s for s, ks in by.items() if any(re.search(pattern_A, n, flags) for _, _, n in ks)]
    hitB = [s for s, ks in by.items() if any(re.search(pattern_B, n, flags) for _, _, n in ks)]
    a = hitA[0] if len(hitA) == 1 else None
    b = [s for s in hitB if s != a]
    b = b[0] if len(b) == 1 else None
    return a, b


def analyse(trace_path, stream_A=None, stream_B=None):
    by = load_kernels(trace_path)
    streams = sorted(by)
    if stream_A is None or stream_B is None:
        # pick the two streams with the most kernels
        streams = sorted(streams, key=lambda s: -len(by[s]))
        if len(streams) < 2:
            raise SystemExit(f"need two streams, found {streams}")
        stream_A, stream_B = streams[0], streams[1]

    A = [(s, e) for s, e, _ in by[stream_A]]
    B = [(s, e) for s, e, _ in by[stream_B]]
    uA, uB = _union(A), _union(B)
    spanA = (uA[0][0], uA[-1][1]) if uA else (0, 0)
    spanB = (uB[0][0], uB[-1][1]) if uB else (0, 0)

    task_span_overlap = max(0.0, min(spanA[1], spanB[1]) - max(spanA[0], spanB[0]))
    kernel_overlap = _intersection(A, B)
    busyA, busyB = _length(uA), _length(uB)

    res = dict(
        stream_A=stream_A, stream_B=stream_B,
        n_kernels_A=len(A), n_kernels_B=len(B),
        span_A_us=spanA, span_B_us=spanB,
        task_span_overlap_us=task_span_overlap,
        kernel_overlap_us=kernel_overlap,
        busy_A_us=busyA, busy_B_us=busyB,
        # normalised by the shorter stream so it is bounded by 1
        kernel_overlap_frac_of_shorter=kernel_overlap / max(min(busyA, busyB), 1e-9),
        task_overlap_frac_of_shorter=task_span_overlap / max(
            min(spanA[1] - spanA[0], spanB[1] - spanB[0]), 1e-9),
        # how much of the other side is left running after the first finishes
        tail_A_after_B_us=max(0.0, spanA[1] - spanB[1]),
        tail_B_after_A_us=max(0.0, spanB[1] - spanA[1]),
        gaps_A_us=(spanA[1] - spanA[0]) - busyA,
        gaps_B_us=(spanB[1] - spanB[0]) - busyB,
        kernel_names_A=sorted({n for _, _, n in by[stream_A]})[:5],
        kernel_names_B=sorted({n for _, _, n in by[stream_B]})[:5],
    )
    return res


if __name__ == "__main__":
    import sys
    print(json.dumps(analyse(sys.argv[1]), indent=2))
