"""Figures for the v3 audit: protocol comparison, physical overlap, corrected S_A."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

summ = json.load(open("raw/gate1_summary.json"))
tl = json.load(open("raw/timeline_overlap.json"))

labels = [f"{s['M']}\n{s['dtype']}" for s in summ]
x = np.arange(len(summ))
Sg = [s["S_A_graph"] for s in summ]
Sl = [s["S_A_legacy"] for s in summ]

# --- fig 1: graph vs legacy S_A + start skew -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(x - 0.2, Sg, 0.4, label="CUDA Graph protocol", color="C0")
ax1.bar(x + 0.2, Sl, 0.4, label="legacy two-stream", color="C1")
ax1.axhline(1.0, color="k", lw=0.8, ls="--")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("S_A (main-task slowdown)")
ax1.set_title("Same condition, two protocols")
ax1.legend(fontsize=8); ax1.grid(alpha=.3, axis="y")

sk_g = [s["skew_graph_ms"] for s in summ]
sk_l = [s["skew_legacy_ms"] for s in summ]
ax2.bar(x - 0.2, sk_g, 0.4, label="graph", color="C0")
ax2.bar(x + 0.2, sk_l, 0.4, label="legacy", color="C1")
ax2.set_yscale("log")
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("device start skew (ms, log)")
ax2.set_title("How far apart did the two tasks actually start?")
ax2.legend(fontsize=8); ax2.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("figures/fig_v3_protocol.png", dpi=140); plt.close()

# --- fig 2: physical kernel overlap ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
lbl = [f"{r['M']}\n{r['precision']}" for r in tl]
xs = np.arange(len(tl))
ovl = [100 * r["kernel_overlap_frac_of_shorter"] for r in tl]
spn = [100 * r["task_overlap_frac_of_shorter"] for r in tl]
ax.bar(xs - 0.2, spn, 0.4, label="task-span overlap", color="C7")
ax.bar(xs + 0.2, ovl, 0.4, label="kernel overlap (physical)", color="C2")
ax.set_xticks(xs); ax.set_xticklabels(lbl, fontsize=8)
ax.set_ylabel("% of the shorter task's busy time")
ax.set_title("Physical kernel overlap from GPU timeline (under profiler)")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("figures/fig_v3_overlap.png", dpi=140); plt.close()

# --- fig 3: corrected S_A vs M -------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for dt, mk in (("bf16", "o"), ("fp4", "s")):
    ms = [s["M"] for s in summ if s["dtype"] == dt]
    va = [s["S_A_graph"] for s in summ if s["dtype"] == dt]
    ax.plot(ms, va, marker=mk, label=f"{dt} main / bf16 tensor probe")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xlabel("M")
ax.set_ylabel("S_A (corrected protocol)")
ax.set_title("Tensor-probe interference, CUDA Graph protocol (k=1)")
ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/fig_v3_sA.png", dpi=140); plt.close()
print("figures written")
