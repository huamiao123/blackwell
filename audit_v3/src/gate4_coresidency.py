"""Gate 4: co-residency probe (falsification test for hypothesis H3).

H3 says: two full-grid kernels cannot be co-resident on 170 SMs, so what we call
"concurrency" is actually kernel-granularity time slicing. The overlap numbers are
consistent with that (28.5% at M=1024/BF16 while S_A is the lowest of all points).

Test: shrink the probe's grid to k CTAs (k = 1, 4, 16, 64, 170) while keeping the
main task at full grid, and measure
    * physical kernel_overlap from a GPU timeline
    * S_A from events
If overlap rises monotonically with k -> the limit is SM availability, H3 survives.
If overlap stays low even at k=1 -> something else (scheduler, per-CTA SMEM) blocks
co-residency and H3 needs revision.

Writes raw/gate4.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import concurrent_graph as CG
import timeline_analysis as TA
from interference_v3 import main_gemm
from probe_fp32_fma import FmaProbe

M = 1024
out = []
rng = np.random.default_rng(0)
fA = main_gemm(M, 12288, 4096, "bf16", rng)
tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
rA = max(5, int(round(20.0 / tA1)))
gA = CG.build_graph(fA, rA)
alone_A = CG.measure_alone(gA)["alone_ms"]

for k in (1, 4, 16, 64, 170):
    # size the FMA probe so its duration is roughly the main task's
    probe = FmaProbe(n_blocks=k, block=256, reps=2000)
    t1 = CG.measure_alone(CG.build_graph(probe, 1))["alone_ms"]
    rB = max(2, int(round(alone_A / max(t1, 1e-6))))
    gB = CG.build_graph(probe, rB)
    c = CG.run_concurrent(gA, gB, "AB")
    trace = f"raw/trace_gate4_k{k}.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        CG.run_concurrent(gA, gB, "AB")
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    # identify streams by kernel name, not by kernel count: the main GEMM kernels
    # contain "gemm"/"cutlass", the Triton FMA probe kernel is "_fma_kernel"
    sA_id, sB_id = TA.streams_by_kernel_pattern(trace, r"gemm|cutlass", r"fma")
    tl = TA.analyse(trace, stream_A=sA_id, stream_B=sB_id)
    row = dict(ctas=k, reps_B=rB, alone_A_ms=alone_A,
               S_A=c["TA_conc_ms"] / alone_A, start_skew_ms=c["start_skew_ms"],
               kernel_overlap_frac=tl["kernel_overlap_frac_of_shorter"],
               task_overlap_frac=tl["task_overlap_frac_of_shorter"],
               busy_B_us=tl["busy_B_us"])
    out.append(row)
    print(f"ctas={k:4d}  S_A={row['S_A']:.3f}  kernel_ovl={100*row['kernel_overlap_frac']:5.1f}%  "
          f"task_ovl={100*row['task_overlap_frac']:5.1f}%  skew={row['start_skew_ms']:.3f} ms")
    del probe, gB
    torch.cuda.empty_cache()

json.dump(out, open("raw/gate4.json", "w"), indent=2)
print("json -> raw/gate4.json")
