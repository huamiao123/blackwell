"""Capture a GPU timeline for the concurrent window and extract physical overlap.

Stream identity fix (review finding): the previous version let `analyse()` guess
which stream was which by kernel count. That is wrong - at M=1024/BF16 the probe
stream had 195 kernels and the main GEMM stream 41, so the two were swapped and
busy-time / tail attribution was inverted.

Now the streams are created once and their ids are learned by profiling each task
*alone* on the very same streams before the concurrent capture. No guessing.

Caveat: CUPTI perturbs execution. Overlap numbers are a cross-check, not the
primary timing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import concurrent_graph as CG
import timeline_analysis as TA
from interference_v3 import main_gemm
from probe_tensor import TensorProbe


def streams_in(path):
    return sorted(TA.load_kernels(path).keys())


def capture_alone(graph, path, stream):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)


def one(M, prec, k=1.0):
    rng = np.random.default_rng(0)
    fA = main_gemm(M, 12288, 4096, prec, rng)
    tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
    rA = max(5, int(round(20.0 / tA1)))
    gA = CG.build_graph(fA, rA)

    probe = TensorProbe("bf16", 2048)
    tB1 = CG.measure_alone(CG.build_graph(probe, 1))["alone_ms"]
    rB = max(2, int(round(k * tA1 * rA / max(tB1, 1e-6))))
    gB = CG.build_graph(probe, rB)

    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()

    # learn the stream ids from single-task captures on the same streams
    pa, pb = f"raw/_alone_A_M{M}_{prec}.json", f"raw/_alone_B_M{M}_{prec}.json"
    capture_alone(gA, pa, sA)
    capture_alone(gB, pb, sB)
    sa_ids, sb_ids = streams_in(pa), streams_in(pb)
    stream_A = sa_ids[0] if sa_ids else None
    stream_B = next((s for s in sb_ids if s != stream_A), sb_ids[0] if sb_ids else None)

    # steady state, then the measured window
    CG.run_concurrent(gA, gB, "AB", sA, sB)
    torch.cuda.synchronize()
    trace = f"raw/trace_M{M}_{prec}.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        CG.run_concurrent(gA, gB, "AB", sA, sB)
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)

    res = TA.analyse(trace, stream_A=stream_A, stream_B=stream_B)
    res.update(M=M, precision=prec, reps_A=rA, reps_B=rB,
               stream_A_from_alone=stream_A, stream_B_from_alone=stream_B,
               stream_ids_alone_A=sa_ids, stream_ids_alone_B=sb_ids)
    for p in (pa, pb):
        Path(p).unlink(missing_ok=True)
    return res


if __name__ == "__main__":
    out = []
    for M in (16, 128, 1024):
        for prec in ("bf16", "fp4"):
            r = one(M, prec)
            out.append(r)
            print(f"M={M:5d} {prec:>4}  streamA={r['stream_A']} streamB={r['stream_B']}  "
                  f"task_ovl={r['task_span_overlap_us']:9.1f}us "
                  f"kernel_ovl={r['kernel_overlap_us']:9.1f}us "
                  f"({100*r['kernel_overlap_frac_of_shorter']:5.1f}% of shorter) "
                  f"busyA={r['busy_A_us']:9.1f} busyB={r['busy_B_us']:9.1f}")
    json.dump(out, open("raw/timeline_overlap.json", "w"), indent=2)
    print("json -> raw/timeline_overlap.json")
