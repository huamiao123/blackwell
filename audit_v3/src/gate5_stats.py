"""Gate 5: bootstrap confidence intervals + Pareto frontier from Gate 3 raw data.

Bootstrap unit = one session x submission order sample (6 per condition), resampled
2000x. This is honest about the independence structure we actually have: kernels
inside one session are NOT independent samples.

Also writes the corrected-protocol Pareto figure:
  x = P_A / P_A_alone   (main throughput normalised by its own solo throughput)
  y = P_B / P_B_alone   (probe throughput normalised by its own solo throughput)
Normalisation is required because different probes do different amounts of work per
call, so raw calls/s are not comparable across probe types.
"""
import csv
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
d = json.load(open("raw/gate3.json"))

samples = []          # one row per (condition, session, order)
for r in d["results"]:
    for sess in r["raw"]:
        for order in ("AB", "BA"):
            c = sess[f"conc_{order}"]
            samples.append(dict(
                M=r["M"], precision=r["precision"], probe=r["probe"], k=r["k"],
                session=sess["session"], order=order,
                alone_A=sess["alone_A_ms"], alone_B=sess["alone_B_ms"],
                TA=c["TA_conc_ms"], TB=c["TB_conc_ms"], TJ=c["concurrent_ms"],
                serial=sess[f"serial_{order}_ms"],
                reps_A=sess["reps_A"], reps_B=sess["reps_B"]))

for s in samples:
    s["S_A"] = s["TA"] / s["alone_A"]
    s["S_B"] = s["TB"] / s["alone_B"]
    s["G"] = s["serial"] / s["TJ"]


def boot_ci(vals, n=2000, alpha=0.05):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    meds = []
    for _ in range(n):
        pick = rng.choice(len(vals), len(vals), replace=True)
        meds.append(st.median([vals[i] for i in pick]))
    meds.sort()
    return (meds[int(alpha / 2 * n)], meds[int((1 - alpha / 2) * n)])


keys = sorted({(s["M"], s["precision"], s["probe"], s["k"]) for s in samples},
              key=lambda t: (t[2], t[3], t[0], t[1]))
rows = []
for key in keys:
    sel = [s for s in samples if (s["M"], s["precision"], s["probe"], s["k"]) == key]
    for metric in ("S_A", "S_B", "G"):
        vals = [s[metric] for s in sel]
        lo, hi = boot_ci(vals)
        rows.append(dict(M=key[0], precision=key[1], probe=key[2], k=key[3],
                         metric=metric, median=st.median(vals),
                         ci_lo=lo, ci_hi=hi, n=len(vals)))
with open("raw/gate5_ci.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"{'M':>5} {'dt':>5} {'probe':>6} {'k':>4} | {'S_A [95% CI]':>28} | {'G [95% CI]':>26}")
for r in rows:
    if r["metric"] != "S_A":
        continue
    g = [x for x in rows if x["metric"] == "G" and (x["M"], x["precision"], x["probe"], x["k"])
         == (r["M"], r["precision"], r["probe"], r["k"])][0]
    print(f"{r['M']:5d} {r['precision']:>5} {r['probe']:>6} {r['k']:4.1f} | "
          f"{r['median']:6.3f} [{r['ci_lo']:6.3f},{r['ci_hi']:6.3f}] | "
          f"{g['median']:6.3f} [{g['ci_lo']:6.3f},{g['ci_hi']:6.3f}]")

# ---------------- Pareto ----------------
def pareto_points(M, dt, probe):
    pts = []
    for k in (0.5, 1.0, 2.0):
        sel = [s for s in samples if (s["M"], s["precision"], s["probe"], s["k"]) == (M, dt, probe, k)]
        if not sel:
            continue
        pA = 1000 * st.median([s["reps_A"] / s["TJ"] for s in sel])
        pB = 1000 * st.median([s["reps_B"] / s["TJ"] for s in sel])
        nA = st.median([(s["reps_A"] / s["TJ"]) / (s["reps_A"] / s["alone_A"]) for s in sel])
        nB = st.median([(s["reps_B"] / s["TJ"]) / (s["reps_B"] / s["alone_B"]) for s in sel])
        pts.append(dict(k=k, pA=pA, pB=pB, nA=nA, nB=nB))
    return pts


def nondominated(pts):
    out = []
    for p in pts:
        dominated = any((q["nA"] >= p["nA"] and q["nB"] >= p["nB"]) and
                        (q["nA"] > p["nA"] or q["nB"] > p["nB"]) for q in pts)
        out.append(not dominated)
    return out


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, M in zip(axes, (128, 1024)):
    for dt, mk, col in (("bf16", "o", "C0"), ("fp4", "s", "C1")):
        pts = pareto_points(M, dt, "tensor")
        if not pts:
            continue
        xs = [p["nA"] for p in pts]; ys = [p["nB"] for p in pts]
        nd = nondominated(pts)
        ax.plot(xs, ys, "-", color=col, lw=1, alpha=.6)
        for x, y, k, is_nd in zip(xs, ys, [p["k"] for p in pts], nd):
            ax.scatter([x], [y], marker=mk, color=col, s=70 if is_nd else 40,
                       edgecolor="k" if is_nd else "none", zorder=3)
            ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.plot([], [], marker=mk, color=col, ls="-", label=f"{dt} main")
    ax.axhline(0, color="k", lw=.5)
    ax.axvline(0, color="k", lw=.5)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.15)
    ax.set_xlabel("main throughput / its solo throughput")
    ax.set_ylabel("probe throughput / its solo throughput")
    ax.set_title(f"M={M}, tensor probe (thick marker = non-dominated)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig_gate5_pareto.png", dpi=140); plt.close()
print("\nfigures/fig_gate5_pareto.png written")
print("raw/gate5_ci.csv written")
