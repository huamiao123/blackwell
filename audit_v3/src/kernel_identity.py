"""Kernel identity per M: which kernel actually runs for each main task, and does
the choice change with M? Also identifies both probes.

Recorded because a change of kernel across M would make a cross-M comparison
compare implementations, not precisions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from interference_v3 import main_gemm
from probe_tensor import TensorProbe

rows = []


def top_kernel(fn, reps=5):
    for _ in range(3):
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
    return (best.key, best.count, best.device_time_total) if best else (None, 0, 0.0)


for M in (16, 32, 64, 128, 256, 512, 1024):
    for dt in ("bf16", "fp4"):
        rng = np.random.default_rng(0)
        fn = main_gemm(M, 12288, 4096, dt, rng)
        name, cnt, us = top_kernel(fn)
        rows.append(dict(kind="main", M=M, dtype=dt, kernel=name, count=cnt, cuda_us=us))
        print(f"main M={M:5d} {dt:>4}  {us/1000:7.3f} ms  {str(name)[:95]}")
        del fn
        torch.cuda.empty_cache()

for dt in ("bf16", "fp4"):
    p = TensorProbe(dt, 2048)
    name, cnt, us = top_kernel(p)
    rows.append(dict(kind="probe", M=2048, dtype=dt, kernel=name, count=cnt, cuda_us=us))
    print(f"probe        {dt:>4}  {us/1000:7.3f} ms  {str(name)[:95]}")
    del p
    torch.cuda.empty_cache()

json.dump(rows, open("raw/kernel_identity.json", "w"), indent=2)
print("json -> raw/kernel_identity.json")
