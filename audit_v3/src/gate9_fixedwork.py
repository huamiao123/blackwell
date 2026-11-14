"""Task 1 / P0: STRICT FIXED-WORK dual-task experiment.

Differences from every earlier run:
  * the main task executes the SAME number of GEMMs for BF16 and FP4 (rA is shared),
    so FP4 simply finishes its share earlier instead of being padded with extra reps;
  * the auxiliary task executes a FIXED workload (fixed FMA count / fixed GEMM count /
    fixed streamed bytes), identical for both precisions - no time matching.

Equal quantity = work only. Durations differ by design and are reported as such.

Per (M, dtype, probe, level) it records medians over sessions x submission order of:
  T_A_alone, T_B_alone, T_serial_AB/BA, T_joint_AB/BA, S_A, S_B, G, start skew,
  and the actual completed work of both tasks.

Usage: python src/gate9_fixedwork.py [--Ms 16,32,...] [--sessions 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import concurrent_graph as CG
from interference_v3 import main_gemm
from probe_tensor import TensorProbe
from probe_dram import DramProbe
from probe_fp32_fma import FmaProbe

N, K = 12288, 4096

# three fixed workload levels per probe type; identical for BF16 and FP4
LEVELS = {
    "fp32_fma": [250_000, 1_000_000, 2_000_000],   # loop iterations (each = 4 FMA/lane)
    "tensor":   [20, 80, 320],                     # 2048^3 BF16 GEMMs
    "dram":     [2, 8, 32],                        # 1.5 GiB streaming passes
}


def make_probe(kind, units):
    if kind == "fp32_fma":
        return FmaProbe(n_blocks=170, block=256, reps=units), 1
    if kind == "tensor":
        return TensorProbe("bf16", 2048), units
    if kind == "dram":
        return DramProbe(mb=1536), units
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", default="16,32,64,128,256,512,1024")
    ap.add_argument("--M", type=int, default=None, help="run a single M (isolates CUDA context)")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    Ms = [args.M] if args.M is not None else [int(x) for x in args.Ms.split(",")]
    if args.json is None:
        args.json = f"raw/gate9_M{args.M}.json" if args.M is not None else "raw/gate9_fixedwork.json"

    rows = []
    print(f"{'M':>5} {'dt':>5} {'probe':>9} {'lvl':>4} {'rA':>4} {'rB':>4} "
          f"{'TA_ms':>8} {'TB_ms':>8} {'SerAB':>8} {'JntAB':>8} {'JntBA':>8} {'S_A':>5} {'S_B':>5} {'G':>5}")
    for M in Ms:
        # ---- one rA per M, shared by both precisions: identical GEMM count ----
        rng0 = np.random.default_rng(0)
        f_bf = main_gemm(M, N, K, "bf16", rng0)
        t1 = CG.measure_alone(CG.build_graph(f_bf, 1))["alone_ms"]
        rA = max(8, int(round(20.0 / t1)))
        del f_bf
        torch.cuda.empty_cache()

        for dt in ("bf16", "fp4"):
            rng = np.random.default_rng(0)
            fA = main_gemm(M, N, K, dt, rng)
            gA = CG.build_graph(fA, rA)
            alone_A = CG.measure_alone(gA)["alone_ms"]

            for kind, levels in LEVELS.items():
                for li, units in enumerate(levels):
                    probe, rB = make_probe(kind, units)
                    gB = CG.build_graph(probe, rB)
                    rec = dict(alone_A=[], alone_B=[], serAB=[], serBA=[], jAB=[], jBA=[], skew=[])
                    for _ in range(args.sessions):
                        alone_B = CG.measure_alone(gB)["alone_ms"]
                        cAB = CG.run_concurrent(gA, gB, "AB")
                        cBA = CG.run_concurrent(gA, gB, "BA")
                        sAB = CG.run_serial(gA, gB, "AB")["serial_ms"]
                        sBA = CG.run_serial(gA, gB, "BA")["serial_ms"]
                        rec["alone_A"].append(alone_A)
                        rec["alone_B"].append(alone_B)
                        rec["serAB"].append(sAB); rec["serBA"].append(sBA)
                        rec["jAB"].append(cAB); rec["jBA"].append(cBA)
                        rec["skew"].append(cAB["start_skew_ms"])
                        rec["skew"].append(cBA["start_skew_ms"])
                    med = lambda xs: float(np.median(xs))
                    sA = [c["TA_conc_ms"] / a for c, a in
                          ([(c, a) for c, a in zip(rec["jAB"] + rec["jBA"],
                                                   rec["alone_A"] * 2)])]
                    sB = [c["TB_conc_ms"] / b for c, b in zip(rec["jAB"] + rec["jBA"],
                                                              rec["alone_B"] * 2)]
                    g = [s / c["concurrent_ms"] for s, c in
                         zip(rec["serAB"] + rec["serBA"], rec["jAB"] + rec["jBA"])]
                    row = dict(M=M, dtype=dt, probe=kind, level=li + 1, probe_units=units,
                               reps_A=rA, reps_B=rB,
                               main_work_GEMMs=rA,
                               probe_work=(units * 170 * 256 * 4 if kind == "fp32_fma"
                                           else units if kind == "tensor" else units * 1536),
                               T_A_alone_ms=med(rec["alone_A"]),
                               T_B_alone_ms=med(rec["alone_B"]),
                               T_serial_AB_ms=med(rec["serAB"]), T_serial_BA_ms=med(rec["serBA"]),
                               T_joint_AB_ms=med([c["concurrent_ms"] for c in rec["jAB"]]),
                               T_joint_BA_ms=med([c["concurrent_ms"] for c in rec["jBA"]]),
                               S_A=med(sA), S_A_min=min(sA), S_A_max=max(sA),
                               S_B=med(sB), G=med(g), skew_ms=med(rec["skew"]))
                    rows.append(row)
                    print(f"{M:5d} {dt:>5} {kind:>9} {li+1:4d} {rA:4d} {rB:4d} "
                          f"{row['T_A_alone_ms']:8.2f} {row['T_B_alone_ms']:8.2f} "
                          f"{row['T_serial_AB_ms']:8.2f} {row['T_joint_AB_ms']:8.2f} "
                          f"{row['T_joint_BA_ms']:8.2f} {row['S_A']:5.2f} {row['S_B']:5.2f} {row['G']:5.2f}")
                    del probe, gB
                    torch.cuda.empty_cache()
            del gA, fA
            torch.cuda.empty_cache()

    json.dump(dict(levels=LEVELS, sessions=args.sessions, rows=rows), open(args.json, "w"), indent=2)
    print("json ->", args.json)


if __name__ == "__main__":
    main()
