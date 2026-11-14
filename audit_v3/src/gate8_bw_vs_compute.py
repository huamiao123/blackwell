"""Gate 8: is FP4's extra interference sensitivity compute-side or bandwidth-side?

The Gate 3 "tensor probe" was a 2048^3 BF16 GEMM. That is NOT bandwidth-free:
A and B are 8 MB each, so one call moves ~16 MB in ~100 us -> ~160 GB/s of DRAM
traffic. So when FP4 slowed down more than BF16 under that probe, we could not
tell whether the cause was the tensor pipe or the memory system.

This script separates the two:

  probe_compute_tensor : 512^3 BF16 GEMM. Operands are 0.5 MB each -> both fit in
                         L2 (96 MiB) and stay there across repetitions, so DRAM
                         traffic is ~0 while the tensor pipe is fully exercised.
  probe_dram           : streaming read+write over 1.5 GiB, arithmetic intensity
                         ~1 FMA / 8 B -> bandwidth dominated, ALU idle.
  probe_compute_scalar : FP32 FMA loop, constant footprint -> scalar pipe only.

Decision rule (stated before running):
  * FP4 penalty appears with the L2-resident tensor probe  -> compute/tensor side
  * FP4 penalty appears only with the DRAM probe           -> bandwidth side
  * both                                                   -> mixed; compare magnitudes

Also records each probe's footprint so the bandwidth argument is explicit.
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
from probe_dram import DramProbe
from probe_fp32_fma import FmaProbe

PROBES = {
    # name: (factory, footprint_bytes, note)
    "tensor_L2": (lambda: TensorProbe("bf16", 512), 2 * 512 * 512 * 2, "512^3 BF16 GEMM, operands fit L2"),
    "tensor_dram": (lambda: TensorProbe("bf16", 2048), 2 * 2048 * 2048 * 2, "2048^3 BF16 GEMM, operands > L2"),
    "dram": (lambda: DramProbe(mb=1536), 2 * 1536 * 1024 * 1024, "1.5 GiB streaming copy"),
    "fp32_fma": (lambda: FmaProbe(n_blocks=170, block=256, reps=2000), 170 * 256 * 4 * 2, "FP32 FMA loop, constant footprint"),
}

out = []
for M in (128, 1024):
    for dt in ("bf16", "fp4"):
        rng = np.random.default_rng(0)
        fA = main_gemm(M, 12288, 4096, dt, rng)
        tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
        rA = max(5, int(round(20.0 / tA1)))
        gA = CG.build_graph(fA, rA)
        alone_A = CG.measure_alone(gA)["alone_ms"]
        for pname, (pf, foot, note) in PROBES.items():
            recs = one_condition(fA, rA, gA, pf, 1.0, 3, ["graph"])
            sA = [r[f"conc_{o}"]["TA_conc_ms"] / r["alone_A_ms"] for r in recs for o in ("AB", "BA")]
            sB = [r[f"conc_{o}"]["TB_conc_ms"] / r["alone_B_ms"] for r in recs for o in ("AB", "BA")]
            # implied DRAM demand of the probe alone
            tB = float(np.median([r["alone_B_ms"] for r in recs])) / 1000.0
            row = dict(M=M, dtype=dt, probe=pname, footprint_bytes=foot, note=note,
                       alone_A_ms=alone_A,
                       S_A_median=float(np.median(sA)), S_A_min=min(sA), S_A_max=max(sA),
                       S_B_median=float(np.median(sB)),
                       probe_ms=float(np.median([r["alone_B_ms"] for r in recs])),
                       n=len(sA))
            out.append(row)
            print(f"M={M:5d} {dt:>4} {pname:>11}  S_A={row['S_A_median']:.3f} "
                  f"[{row['S_A_min']:.3f},{row['S_A_max']:.3f}]  S_B={row['S_B_median']:.3f} "
                  f"| footprint={foot/1e6:.1f} MB")
        del gA, fA
        torch.cuda.empty_cache()

json.dump(dict(probes={k: dict(footprint_bytes=v[1], note=v[2]) for k, v in PROBES.items()},
               results=out), open("raw/gate8.json", "w"), indent=2)
print("json -> raw/gate8.json")
