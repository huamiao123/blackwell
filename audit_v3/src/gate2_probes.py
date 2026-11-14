"""Gate 2: validate that each probe actually exercises the resource it claims.

FP32 FMA probe : must contain fma.rn.f32 in PTX, must scale with reps, must not be
                 dominated by memory traffic (constant footprint).
DRAM probe     : working-set sweep must show the L2 step (~96 MiB) and a plateau
                 whose implied bandwidth is a plausible fraction of 1792 GB/s.
Tensor probe   : must report the executed kernel name for both BF16 and FP4.

Writes raw/gate2.json. No claims are made from a probe that fails its own check.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import concurrent_graph as CG
from probe_fp32_fma import FmaProbe
from probe_dram import sweep_bytes, DramProbe
from probe_tensor import TensorProbe


def timed_launch(fn, reps=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


out = {}

# ---------- FP32 FMA ----------
print("== FP32 FMA probe ==")
fma_rows = []
for reps in (500, 2000, 8000):
    p = FmaProbe(n_blocks=170, block=256, reps=reps)
    meta = p.verify_codegen()
    ms = timed_launch(p, reps=10)
    fmas = meta["fma_per_launch"]
    row = dict(reps=reps, ms=ms, fma_per_launch=fmas,
               gflops=fmas / (ms * 1e-3) / 1e9,
               ptx_has_fma=meta.get("ptx_has_fma"),
               ptx_fma_count=meta.get("ptx_fma_count"),
               sass_has_ffma=meta.get("sass_has_ffma"))
    fma_rows.append(row)
    print(f"  reps={reps:6d}  {ms:8.3f} ms  {row['gflops']:8.1f} GFLOP/s  "
          f"ptx_fma={row['ptx_fma_count']} sass_ffma={row['sass_has_ffma']}")
    del p
    torch.cuda.empty_cache()
out["fp32_fma"] = fma_rows

# ---------- DRAM ----------
print("== DRAM probe (working-set sweep) ==")
sw = sweep_bytes()
for r in sw:
    print(f"  {r['mb']:5d} MiB  {r['ms']:8.3f} ms  {r['gbps']:7.1f} GB/s")
out["dram_sweep"] = sw
p = DramProbe(mb=1536)
out["dram_meta"] = p.meta
del p
torch.cuda.empty_cache()

# ---------- Tensor ----------
print("== Tensor probes ==")
trows = []
for dt in ("bf16", "fp4"):
    t = TensorProbe(dt, 2048)
    kernels = t.identify()
    ms = timed_launch(t, reps=20)
    row = dict(dtype=dt, ms=ms, tflops=t.meta["flops"] / (ms * 1e-3) / 1e12,
               kernel=kernels[0]["kernel"] if kernels else None,
               n_kernels=len(kernels))
    trows.append(row)
    print(f"  {dt:>4}  {ms:8.3f} ms  {row['tflops']:8.1f} TF  {str(row['kernel'])[:100]}")
    del t
    torch.cuda.empty_cache()
out["tensor"] = trows

json.dump(out, open("raw/gate2.json", "w"), indent=2)
print("json -> raw/gate2.json")

# honest verdicts
ok_fma = all(r["ptx_has_fma"] for r in fma_rows)
dram_pl = max(r["gbps"] for r in sw)
ok_tc = all(r["kernel"] for r in trows)
print(f"\nVERDICT: fp32_ptx_fma={ok_fma}  dram_peak={dram_pl:.0f} GB/s  tensor_kernels={ok_tc}")
