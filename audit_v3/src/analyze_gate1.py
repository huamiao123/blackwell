"""Gate-1 analysis: corrected (graph) vs legacy (two-stream) protocol.

The legacy protocol used eager launches, so its slowdown must be normalised by an
EAGER alone baseline; normalising it by the graph-based baseline would mix two
effects. Both baselines are recorded, and only the eager-normalised legacy number
is reported as the legacy result.

Prints per condition:
  S_A / S_B / G under the corrected protocol
  S_A / S_B under the legacy protocol (eager baseline)
  host submission time and device start skew for both
"""
import json
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "raw/gate1.json"
d = json.load(open(path))


def med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


rows = []
for r in d["results"]:
    for sess in r["raw"]:
        aA, aB = sess["alone_A_ms"], sess["alone_B_ms"]
        eA = sess.get("alone_A_eager_ms")
        eB = sess.get("alone_B_eager_ms")
        row = dict(M=r["M"], dtype=r["precision"], k=r["k"], session=sess["session"],
                   k_realised=sess["k_realised"], alone_graph=aA, alone_eager=eA)
        for order in ("AB", "BA"):
            c = sess[f"conc_{order}"]
            s = sess[f"serial_{order}_ms"]
            row[f"S_A_gr_{order}"] = c["TA_conc_ms"] / aA
            row[f"S_B_gr_{order}"] = c["TB_conc_ms"] / aB
            row[f"G_gr_{order}"] = s / c["concurrent_ms"]
            row[f"skew_{order}"] = c["start_skew_ms"]
            row[f"submit_{order}"] = c["cpu_submit_ms"]
            lg = sess.get(f"legacy_{order}")
            if lg and eA and eB:
                row[f"S_A_lg_{order}"] = lg["TA_conc_ms"] / eA
                row[f"S_B_lg_{order}"] = lg["TB_conc_ms"] / eB
                row[f"skew_lg_{order}"] = lg["start_skew_ms"]
                row[f"submit_lg_{order}"] = lg["cpu_submit_ms"]
        rows.append(row)

keys = sorted({(r["M"], r["dtype"]) for r in rows})
hdr = (f"{'M':>5} {'dt':>5} | {'S_A graph':>15} {'S_A legacy':>15} {'delta':>7} | "
       f"{'skew gr AB':>10} {'skew gr BA':>10} | {'skew lg AB':>10} {'skew lg BA':>10}")
print(hdr); print("-" * len(hdr))
summary = []
for M, dt in keys:
    sel = [r for r in rows if r["M"] == M and r["dtype"] == dt]
    g = [r[f"S_A_gr_{o}"] for r in sel for o in ("AB", "BA")]
    l = [r.get(f"S_A_lg_{o}") for r in sel for o in ("AB", "BA") if r.get(f"S_A_lg_{o}")]
    gb = [r[f"S_B_gr_{o}"] for r in sel for o in ("AB", "BA")]
    lb = [r.get(f"S_B_lg_{o}") for r in sel for o in ("AB", "BA") if r.get(f"S_B_lg_{o}")]
    skAB = [r["skew_AB"] for r in sel]
    skBA = [r["skew_BA"] for r in sel]
    slAB = [r.get("skew_lg_AB") for r in sel if r.get("skew_lg_AB")]
    slBA = [r.get("skew_lg_BA") for r in sel if r.get("skew_lg_BA")]
    sub = [r[f"submit_{o}"] for r in sel for o in ("AB", "BA")]
    subl = [r.get(f"submit_lg_{o}") for r in sel for o in ("AB", "BA") if r.get(f"submit_lg_{o}")]
    mg, ml = med(g), med(l)
    print(f"{M:5d} {dt:>5} | {mg:8.3f}±{(max(g)-min(g))/2:5.3f} {ml:8.3f}±{(max(l)-min(l))/2:5.3f} "
          f"{ml-mg:+7.3f} | {med(skAB):10.3f} {med(skBA):10.3f} | {med(slAB):10.3f} {med(slBA):10.3f}")
    summary.append(dict(M=M, dtype=dt, S_A_graph=mg, S_A_legacy=ml, delta=ml - mg,
                        S_B_graph=med(gb), S_B_legacy=med(lb),
                        skew_graph_AB_ms=med(skAB), skew_graph_BA_ms=med(skBA),
                        skew_legacy_AB_ms=med(slAB), skew_legacy_BA_ms=med(slBA),
                        skew_graph_max_ms=max(skAB + skBA),
                        submit_graph_ms=med(sub), submit_legacy_ms=med(subl),
                        alone_graph_ms=med([r["alone_graph"] for r in sel]),
                        alone_eager_ms=med([r["alone_eager"] for r in sel])))

print()
print("k realised (T_B/T_A), median:", round(med([r["k_realised"] for r in rows]), 3))
print("graph/eager alone ratio, median:",
      round(med([r["alone_graph"] / r["alone_eager"] for r in rows if r["alone_eager"]]), 3))
json.dump(summary, open("raw/gate1_summary.json", "w"), indent=2)
print("summary -> raw/gate1_summary.json")
