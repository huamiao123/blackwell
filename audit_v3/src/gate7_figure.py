"""Figure for Gate 7: S_A vs M, markers grouped by the kernel that actually ran.

Purpose: show whether the BF16 downward trend survives when the kernel is fixed.
The 128x64_64x3 group covers M = 32, 64, 256, 512 - four points on ONE kernel.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("raw/gate7.json"))


def short(k):
    if not k:
        return "?"
    m = re.search(r"(wmma_tensorop_bf16_s16\d+gemm|s16\d+gemm_relu_bf16)(?:_bf16)?[._](\d+x\d+[._]\d+x\d+)", k)
    if m:
        return ("WMMA " if "wmma" in m.group(1) else "") + m.group(2).replace(".", "_")
    m = re.search(r"block_scaled_\w+_(\d+x\d+x\d+)", k)
    return "FP4 " + m.group(1) if m else k[:30]


labels = {}
for r in rows:
    labels.setdefault((r["dtype"], short(r["kernel"])), []).append(r)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
styles = ["o", "s", "^", "D", "v"]
ci = 0
for (dt, kl), rs in sorted(labels.items()):
    rs = sorted(rs, key=lambda r: r["M"])
    xs = [r["M"] for r in rs]
    ys = [r["S_A_median"] for r in rs]
    lo = [r["S_A_median"] - r["S_A_min"] for r in rs]
    hi = [r["S_A_max"] - r["S_A_median"] for r in rs]
    color = "C0" if dt == "bf16" else "C1"
    mk = styles[ci % len(styles)]
    ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, color=color, capsize=3,
                label=f"{dt}: {kl}", lw=1.2 if len(xs) > 1 else 0, ms=7)
    ci += 1

ax.axhline(1.0, color="k", lw=.8, ls="--")
ax.set_xscale("log", base=2); ax.set_xlabel("M")
ax.set_ylabel("S_A (tensor probe, k=1)")
ax.set_title("S_A vs M, grouped by the kernel that actually ran")
ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig("figures/fig_gate7_kernel_groups.png", dpi=140); plt.close()
print("figures/fig_gate7_kernel_groups.png written")
