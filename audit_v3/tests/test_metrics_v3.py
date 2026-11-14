"""CPU-only unit tests for the v3 metric definitions and the v3 estimators.

No GPU is touched. Run:  python tests/test_metrics_v3.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import timeline_analysis as TA

fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (("  " + extra) if extra else ""))
    if not cond:
        fails += 1


def metrics(alone_A, alone_B, conc_A, conc_B, joint, serial):
    return dict(S_A=conc_A / alone_A, S_B=conc_B / alone_B, G=serial / joint)


# --- 1. definitions ---------------------------------------------------------
m = metrics(10.0, 20.0, 20.0, 20.0, 25.0, 30.0)
check("S_A = T_A_conc / T_A_alone", abs(m["S_A"] - 2.0) < 1e-12, f"{m['S_A']}")
check("S_B = T_B_conc / T_B_alone", abs(m["S_B"] - 1.0) < 1e-12, f"{m['S_B']}")
check("G   = T_serial / T_joint", abs(m["G"] - 30.0 / 25.0) < 1e-12, f"{m['G']}")

# --- 2. G must never be built from a summed serial baseline ----------------
serial_sum = 10.0 + 20.0                      # the v2 mistake
check("summed serial != measured serial (guARDS the v2 bug)",
      abs(serial_sum - 30.0) < 1e-12 and m["G"] == 30.0 / 25.0,
      "documented: v2 used tSer = timed(fA)+timed(fB)")

# --- 3. perfect-overlap boundary -------------------------------------------
perf = metrics(10.0, 20.0, 10.0, 20.0, 20.0, 30.0)
check("perfect overlap: joint == max(alone), G == 1.5",
      abs(perf["G"] - 1.5) < 1e-12 and abs(perf["S_A"] - 1.0) < 1e-12)

# --- 4. fully serialised boundary ------------------------------------------
ser = metrics(10.0, 20.0, 10.0, 20.0, 30.0, 30.0)
check("fully serialised: G == 1.0", abs(ser["G"] - 1.0) < 1e-12)

# --- 5. savings metric must NOT be called overlap --------------------------
def swsf(tA, tB, tJ):
    hi, lo = max(tA, tB), min(tA, tB)
    return 0.0 if lo <= 0 else max(0.0, min(1.0, 1.0 - (tJ - hi) / lo))
check("SWSF(serial)=0, SWSF(perfect)=1",
      abs(swsf(10, 20, 30)) < 1e-12 and abs(swsf(10, 20, 20) - 1.0) < 1e-12)
check("SWSF is bounded to [0,1] even with a noisy joint",
      0.0 <= swsf(10, 20, 5) <= 1.0)

# --- 6. timeline interval algebra ------------------------------------------
check("union length", abs(TA._length(TA._union([(0, 10), (10, 20), (30, 40)])) - 30) < 1e-9)
check("intersection of disjoint = 0",
      TA._intersection([(0, 10)], [(20, 30)]) == 0)
check("intersection partial = 5",
      abs(TA._intersection([(0, 10)], [(5, 30)]) - 5) < 1e-9)
check("intersection with repetition counted once",
      abs(TA._intersection([(0, 10)], [(2, 3), (4, 6), (20, 21)]) - 3) < 1e-9)

# --- 7. synthetic trace: task-span overlap > kernel overlap ----------------
# A: kernels [0,10] and [190,200]  -> task span [0,200]
# B: kernel  [100,110]             -> task span [100,110]
# The two TASKS overlap by 10 us (their spans intersect) but their KERNELS
# never run at the same instant. This is exactly the confusion we must avoid.
trace = {"traceEvents": [
    {"cat": "kernel", "ts": 0,   "dur": 10, "name": "A1", "args": {"stream": 1}},
    {"cat": "kernel", "ts": 190, "dur": 10, "name": "A2", "args": {"stream": 1}},
    {"cat": "kernel", "ts": 100, "dur": 10, "name": "B1", "args": {"stream": 2}},
]}
tmp = Path("_tmp_trace.json")
tmp.write_text(json.dumps(trace))
r = TA.analyse(str(tmp))
tmp.unlink()
check("task span overlap = 10 us (spans [0,200] and [100,110])",
      abs(r["task_span_overlap_us"] - 10) < 1e-9, f"{r['task_span_overlap_us']}")
check("kernel overlap = 0 us although the task spans intersect",
      abs(r["kernel_overlap_us"]) < 1e-9, f"{r['kernel_overlap_us']}")
check("task overlap frac of shorter = 1.0 while kernel frac = 0.0",
      abs(r["task_overlap_frac_of_shorter"] - 1.0) < 1e-9
      and abs(r["kernel_overlap_frac_of_shorter"]) < 1e-9)

# --- 8. identical intervals give full kernel overlap ----------------------
trace2 = {"traceEvents": [
    {"cat": "kernel", "ts": 0, "dur": 50, "name": "A", "args": {"stream": 1}},
    {"cat": "kernel", "ts": 0, "dur": 50, "name": "B", "args": {"stream": 2}},
]}
tmp2 = Path("_tmp_trace2.json"); tmp2.write_text(json.dumps(trace2))
r2 = TA.analyse(str(tmp2)); tmp2.unlink()
check("identical intervals -> kernel overlap == busy time",
      abs(r2["kernel_overlap_frac_of_shorter"] - 1.0) < 1e-9)

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
