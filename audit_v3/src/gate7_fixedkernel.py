"""Gate 7: does the BF16 S_A(M) trend survive when the kernel is held fixed?

The per-M kernel identity table (raw/kernel_identity.json) showed BF16 switching
between WMMA s161616 (M=16), 128x64_64x3 (32/64/256/512), 256x64_32x4 (128) and
64x256_32x4 (1024). So the earlier S_A-vs-M curve mixed kernel changes with M.

This script measures S_A per M *and records the kernel identity in the same
process*, so the analysis can group by kernel instead of by M.

Output: raw/gate7.json  (rows: M, kernel, S_A samples, S_B, G, skew)
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
from interference_v3 import main_gemm, one_condition
from probe_tensor import TensorProbe


def top_kernel(fn, reps=3):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    best = None
    for e in prof.key_averages():
        if e.device_time_total > 0 and (best is None or e.device_time_total > best.device_time_total):
            best = e
    return best.key if best else None


out = []
for M in (16, 32, 64, 128, 256, 512, 1024):
    for dt in ("bf16", "fp4"):
        rng = np.random.default_rng(0)
        fA = main_gemm(M, 12288, 4096, dt, rng)
        kname = top_kernel(fA)
        tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
        rA = max(5, int(round(20.0 / tA1)))
        gA = CG.build_graph(fA, rA)
        pf = (lambda: TensorProbe("bf16", 2048))
        recs = one_condition(fA, rA, gA, pf, 1.0, 3, ["graph"])
        sA = [r["conc_AB"]["TA_conc_ms"] / r["alone_A_ms"] for r in recs] + \
             [r["conc_BA"]["TA_conc_ms"] / r["alone_A_ms"] for r in recs]
        sB = [r["conc_AB"]["TB_conc_ms"] / r["alone_B_ms"] for r in recs] + \
             [r["conc_BA"]["TB_conc_ms"] / r["alone_B_ms"] for r in recs]
        g = [r[f"serial_{o}_ms"] / r[f"conc_{o}"]["concurrent_ms"] for r in recs for o in ("AB", "BA")]
        row = dict(M=M, dtype=dt, kernel=kname, reps_A=rA,
                   S_A_median=float(np.median(sA)), S_A_min=min(sA), S_A_max=max(sA),
                   S_B_median=float(np.median(sB)), G_median=float(np.median(g)),
                   n=len(sA))
        out.append(row)
        short = kname.split("<")[-1][:60] if "<" in kname else kname[:60]
        print(f"M={M:5d} {dt:>4}  S_A={row['S_A_median']:.3f}  {short}")
        del gA, fA
        torch.cuda.empty_cache()

json.dump(out, open("raw/gate7.json", "w"), indent=2)
print("json -> raw/gate7.json")
