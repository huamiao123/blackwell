"""Interference experiment v3 (corrected protocol).

Two distinct experiments, never mixed:
  * fixed_work      : probe repetition count is fixed across M / precision.
                      What varies is how much of the main task it covers.
  * fixed_duration  : probe reps are calibrated per (main task, k) so that
                      T_B_alone ~= k * T_A_alone.  The realised ratio is stored.

Per condition, per session we record raw samples for:
  alone_A, alone_B (measured fresh), serial AB, serial BA, concurrent AB, concurrent BA,
  plus the legacy two-stream submission for skew comparison.

Run (Gate 1):
    python src/interference_v3.py --gate1 --json raw/gate1.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

import concurrent_graph as CG
from probe_tensor import TensorProbe
from probe_fp32_fma import FmaProbe
from probe_dram import DramProbe


def blocked(s):
    r, c = s.shape
    rb, cb = (r + 127) // 128, (c + 3) // 4
    p = np.zeros((rb * 128, cb * 4), np.float32); p[:r, :c] = s
    return p.reshape(rb, 128, cb, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3).reshape(-1)


def main_gemm(M, N, K, dtype, rng):
    if dtype == "bf16":
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        return lambda: A @ W.t()
    ta = torch.from_numpy(rng.integers(0, 256, (M, K // 2), dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
    tw = torch.from_numpy(rng.integers(0, 256, (N, K // 2), dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
    sa = torch.from_numpy(blocked(rng.uniform(.01, .08, (M, K // 16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
    sb = torch.from_numpy(blocked(rng.uniform(.01, .08, (N, K // 16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
    return lambda: torch._scaled_mm(ta, tw.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)


def gpu_state():
    import subprocess
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=5).decode().strip()
        p, c, t = [float(v) for v in o.split(",")]
        return dict(power_w=p, sm_mhz=c, temp_c=t)
    except Exception:
        return dict(power_w=None, sm_mhz=None, temp_c=None)


def one_condition(fA, rA, gA, probe_factory, k, sessions, modes):
    """Return raw per-session records for one (main, probe, k) condition."""
    recs = []
    # calibrate probe reps once on this main task's duration
    tA = CG.measure_alone(gA)["alone_ms"]
    probe = probe_factory()
    tB1 = CG.measure_alone(CG.build_graph(probe, 1))["alone_ms"]
    rB = max(2, int(round(k * tA / max(tB1, 1e-6))))
    gB = CG.build_graph(probe, rB)
    for s in range(sessions):
        st = gpu_state()
        rec = dict(session=s, reps_A=rA, reps_B=rB, k_nominal=k, **st)
        a = CG.measure_alone(gA)
        b = CG.measure_alone(gB)
        rec.update(alone_A_ms=a["alone_ms"], alone_B_ms=b["alone_ms"])
        # eager baselines: the legacy protocol uses eager launches, so its S_A
        # must be normalised by an eager alone measurement, not by the graph one
        rec["alone_A_eager_ms"] = CG.timed_eager(fA, rA)
        rec["alone_B_eager_ms"] = CG.timed_eager(probe, rB)
        rec["k_realised"] = b["alone_ms"] / a["alone_ms"] if a["alone_ms"] else float("nan")
        rec["serial_AB_ms"] = CG.run_serial(gA, gB, "AB")["serial_ms"]
        rec["serial_BA_ms"] = CG.run_serial(gA, gB, "BA")["serial_ms"]
        rec["conc_AB"] = CG.run_concurrent(gA, gB, "AB")
        rec["conc_BA"] = CG.run_concurrent(gA, gB, "BA")
        if "legacy" in modes:
            fA1, fB1 = probe_factory, probe
            rec["legacy_AB"] = CG.run_concurrent_streams(fA1, fB1, rA, rB, "AB")
            rec["legacy_BA"] = CG.run_concurrent_streams(fA1, fB1, rA, rB, "BA")
        recs.append(rec)
        torch.cuda.empty_cache()
    del probe
    del gB
    return recs


def derive(recs):
    """Derived metrics from raw samples. Ratios are computed per session, then summarised."""
    out = []
    for r in recs:
        aA, aB = r["alone_A_ms"], r["alone_B_ms"]
        for order in ("AB", "BA"):
            c = r[f"conc_{order}"]
            s = r[f"serial_{order}_ms"]
            out.append(dict(session=r["session"], order=order,
                            S_A=c["TA_conc_ms"] / aA,
                            S_B=c["TB_conc_ms"] / aB,
                            G=s / c["concurrent_ms"],
                            serial_source="measured",
                            cpu_submit_ms=c["cpu_submit_ms"],
                            start_skew_ms=c["start_skew_ms"],
                            k_realised=r["k_realised"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", action="store_true")
    ap.add_argument("--Ms", default="16,128,1024")
    ap.add_argument("--precisions", default="bf16,fp4")
    ap.add_argument("--probes", default="tensor")
    ap.add_argument("--ks", default="1.0")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--modes", default="graph,legacy")
    ap.add_argument("--probe-dtype", default="bf16")
    ap.add_argument("--json", default="raw/gate1.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N, K = 12288, 4096
    Ms = [int(x) for x in args.Ms.split(",")]
    precs = args.precisions.split(",")
    probes = args.probes.split(",")
    ks = [float(x) for x in args.ks.split(",")]
    modes = args.modes.split(",")

    conditions = [(M, p, pr, k) for M in Ms for p in precs for pr in probes for k in ks]
    random.Random(args.seed).shuffle(conditions)

    results, t_start = [], time.strftime("%Y-%m-%dT%H:%M:%S")
    for M, prec, probe_name, k in conditions:
        rng = np.random.default_rng(0)
        fA = main_gemm(M, N, K, prec, rng)
        tA1 = CG.measure_alone(CG.build_graph(fA, 1))["alone_ms"]
        rA = max(5, int(round(20.0 / tA1)))
        gA = CG.build_graph(fA, rA)
        pf = (lambda: TensorProbe(args.probe_dtype, 2048)) if probe_name == "tensor" else None
        if probe_name == "fp32":
            pf = (lambda: FmaProbe(n_blocks=170, block=256, reps=2000))
        if probe_name == "dram":
            pf = (lambda: DramProbe(mb=1536))
        recs = one_condition(fA, rA, gA, pf, k, args.sessions, modes)
        d = derive(recs)
        results.append(dict(M=M, precision=prec, probe=probe_name, k=k,
                            reps_A=rA, raw=recs, derived=d))
        sA = [x["S_A"] for x in d]
        sB = [x["S_B"] for x in d]
        g = [x["G"] for x in d]
        print(f"M={M:5d} {prec:>4} {probe_name:>6} k={k:.1f}  "
              f"S_A {min(sA):.2f}-{max(sA):.2f}  S_B {min(sB):.2f}-{max(sB):.2f}  "
              f"G {min(g):.2f}-{max(g):.2f}  skew {max(x['start_skew_ms'] for x in d):.3f} ms")
        del gA, fA
        torch.cuda.empty_cache()

    json.dump(dict(started=t_start, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   args=vars(args), results=results),
              open(args.json, "w"), indent=2)
    print("json ->", args.json)


if __name__ == "__main__":
    main()
