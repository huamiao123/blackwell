"""Capture a GPU timeline for the concurrent window and extract physical overlap.

For each (M, precision) it builds the same graphs as interference_v3, runs one
concurrent AB window under torch.profiler, exports the chrome trace, and reports:
  task_span_overlap   : do the two tasks' overall spans intersect
  kernel_overlap      : do their kernels actually execute at the same instant
  tails / gaps        : does one side finish long before the other

Caveat: CUPTI perturbs execution. These numbers are a cross-check, not the
primary timing. Report them as "measured under profiling".
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
sys.path.insert(0, str(HERE))

import concurrent_graph as CG
import timeline_analysis as TA
from interference_v3 import main_gemm
from probe_tensor import TensorProbe


def one(M, prec, k=1.0, sessions=1):
    rng = np.random.default_rng(0)
    fA = main_gemm(M, 12288, 4096, prec, rng)
    tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
    rA = max(5, int(round(20.0 / tA1)))
    gA = CG.build_graph(fA, rA)
    probe = TensorProbe("bf16", 2048)
    tB1 = CG.measure_alone(CG.build_graph(probe, 1))["alone_ms"]
    rB = max(2, int(round(k * tA1 * rA / max(tB1, 1e-6))))
    gB = CG.build_graph(probe, rB)

    # warm up so the trace contains steady-state kernels only
    CG.run_concurrent(gA, gB, "AB")
    torch.cuda.synchronize()

    trace = f"raw/trace_M{M}_{prec}.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        CG.run_concurrent(gA, gB, "AB")
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    res = TA.analyse(trace)
    res.update(M=M, precision=prec, reps_A=rA, reps_B=rB)
    return res


if __name__ == "__main__":
    out = []
    for M in (16, 128, 1024):
        for prec in ("bf16", "fp4"):
            r = one(M, prec)
            out.append(r)
            print(f"M={M:5d} {prec:>4}  task_span_ovl={r['task_span_overlap_us']:9.1f}us "
                  f"kernel_ovl={r['kernel_overlap_us']:9.1f}us "
                  f"({100*r['kernel_overlap_frac_of_shorter']:5.1f}% of shorter) "
                  f"busyA={r['busy_A_us']:9.1f} busyB={r['busy_B_us']:9.1f} "
                  f"tailB_after_A={r['tail_B_after_A_us']:8.1f}")
    json.dump(out, open("raw/timeline_overlap.json", "w"), indent=2)
    print("json -> raw/timeline_overlap.json")
