"""Gate 3 analysis + figures (126 conditions, corrected protocol).

Reads raw/gate3.json, writes:
  raw/gate3_summary.csv           one row per (M, precision, probe, k)
  figures/fig_gate3_heatmap.png   S_A over probe x M, per precision (k=1)
  figures/fig_gate3_SB.png        S_B over probe x M, per precision (k=1)
  figures/fig_gate3_crossover.png S_A(tensor) vs M with min/max over samples
  figures/fig_gate3_pressure.png  S_A vs k for tensor probe
"""
import csv
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("raw/gate3.json"))
rows = []
for r in d["results"]:
    for der in r["derived"]:
        rows.append(dict(M=r["M"], precision=r["precision"], probe=r["probe"], k=r["k"],
                         session=der["session"], order=der["order"],
                         S_A=der["S_A"], S_B=der["S_B"], G=der["G"],
                         skew_ms=der["start_skew_ms"], submit_ms=der["cpu_submit_ms"],
                         k_realised=der["k_realised"]))

def med(xs):
    return st.median(xs) if xs else float("nan")

keys = sorted({(r["M"], r["precision"], r["probe"], r["k"]) for r in rows},
              key=lambda t: (t[2], t[3], t[0], t[1]))
summary = []
for M, p, pr, k in keys:
    sel = [r for r in rows if (r["M"], r["precision"], r["probe"], r["k"]) == (M, p, pr, k)]
    for metric in ("S_A", "S_B", "G"):
        vals = [r[metric] for r in sel]
        summary.append(dict(M=M, precision=p, probe=pr, k=k, metric=metric,
                            median=med(vals), min=min(vals), max=max(vals), n=len(vals),
                            skew_ms=med([r["skew_ms"] for r in sel])))

with open("raw/gate3_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

Ms = sorted({r["M"] for r in rows})
probes = ["fp32", "dram", "tensor"]

def heatmap(metric, fname, title, vmin, vmax):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, dt in zip(axes, ["bf16", "fp4"]):
        Z = np.full((len(probes), len(Ms)), np.nan)
        for i, pr in enumerate(probes):
            for j, M in enumerate(Ms):
                v = [s["median"] for s in summary
                     if s["metric"] == metric and s["probe"] == pr and s["M"] == M
                     and s["precision"] == dt and abs(s["k"] - 1.0) < 1e-6]
                if v:
                    Z[i, j] = v[0]
        im = ax.imshow(Z, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(Ms))); ax.set_xticklabels(Ms, fontsize=8)
        ax.set_yticks(range(len(probes))); ax.set_yticklabels(probes)
        ax.set_xlabel("M"); ax.set_title(f"{dt}: {metric} (k=1)")
        for i in range(len(probes)):
            for j in range(len(Ms)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(); plt.savefig(fname, dpi=140); plt.close()

heatmap("S_A", "figures/fig_gate3_heatmap.png", "main-task slowdown S_A", 0.9, 3.0)
heatmap("S_B", "figures/fig_gate3_SB.png", "probe slowdown S_B", 0.8, 4.0)

# crossover: tensor probe, k=1
fig, ax = plt.subplots(figsize=(6.4, 4))
for dt, mk in (("bf16", "o"), ("fp4", "s")):
    xs, ys, lo, hi = [], [], [], []
    for M in Ms:
        sel = [s for s in summary if s["metric"] == "S_A" and s["probe"] == "tensor"
               and s["M"] == M and s["precision"] == dt and abs(s["k"] - 1.0) < 1e-6]
        if sel:
            xs.append(M); ys.append(sel[0]["median"])
            lo.append(sel[0]["median"] - sel[0]["min"]); hi.append(sel[0]["max"] - sel[0]["median"])
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, capsize=3, label=f"{dt} main")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xlabel("M")
ax.set_ylabel("S_A (tensor probe, k=1)")
ax.set_title("Which precision tolerates a tensor probe better?")
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig_gate3_crossover.png", dpi=140); plt.close()

# pressure sweep for tensor
fig, ax = plt.subplots(figsize=(6.4, 4))
for dt, mk in (("bf16", "o"), ("fp4", "s")):
    for M, ls in ((128, "-"), (1024, "--")):
        xs, ys = [], []
        for k in (0.5, 1.0, 2.0):
            sel = [s for s in summary if s["metric"] == "S_A" and s["probe"] == "tensor"
                   and s["M"] == M and s["precision"] == dt and abs(s["k"] - k) < 1e-6]
            if sel:
                xs.append(k); ys.append(sel[0]["median"])
        if xs:
            ax.plot(xs, ys, ls, marker=mk, label=f"{dt} M={M}")
ax.set_xlabel("nominal k = T_B/T_A"); ax.set_ylabel("S_A")
ax.set_title("tensor probe: slowdown vs pressure")
ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig_gate3_pressure.png", dpi=140); plt.close()

print("rows:", len(rows), "figure files written")
print("\nS_A (tensor, k=1) crossover table:")
for M in Ms:
    line = [f"M={M:5d}"]
    for dt in ("bf16", "fp4"):
        sel = [s for s in summary if s["metric"] == "S_A" and s["probe"] == "tensor"
               and s["M"] == M and s["precision"] == dt and abs(s["k"] - 1.0) < 1e-6]
        line.append(f"{dt}={sel[0]['median']:.3f}" if sel else f"{dt}=NA")
    print("  " + "  ".join(line))
