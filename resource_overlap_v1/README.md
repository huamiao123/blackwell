# Native NVFP4 Decode: resource-overlap study

This directory extends `../nsys_round2` and `../audit_v3`; it does not replace
their measurements or treat utilization as spare capacity.

Read [methodology.md](methodology.md), [analysis/gate0.json](analysis/gate0.json),
and [analysis/failures.md](analysis/failures.md) before using the scripts.
Current stage and measured results: [final_report.md](final_report.md).
Gate 0 and 20 fair baseline trials are complete. New node traces, probe GPU
correctness and overlap remain pending: another user's job closed the idle
window. No automatic resume is scheduled.

All GPU programs must run through `src/run_supervised.py` during an authorized
idle window. Each invocation launches **one finite gate experiment** and has a
hard timeout; it never queues another gate. New local working files live on D:.
Remote root: `~/fp4-llm/results/resource_overlap_v1/`.

Example Gate 1 (not an instruction to start without checking the shared GPU):

```bash
python3 ~/fp4-llm/results/resource_overlap_v1/src/run_supervised.py \
  --label gate1_nvfp4_b8_unique --timeout 420 -- \
  --precision nvfp4 --batch 8 --steps 256 --trials 5
```

Gate 2 changes `--target probe` and passes `--kind fp32|dram|tensor` with fixed
work parameters. Gate 3 uses `--probe fp32 --chunks 32 --reps 200000 --steps 32`
and separately profiles one `--modes parallel_ab --trials 1` invocation using
`--profile node`. Do not treat a 32-step pilot as a 256-step formal result.

Formal paired modes are
`alone_a,alone_b,serial_ab,serial_ba,parallel_ab,parallel_ba,empty_ab,empty_ba`.
After each gate inspect the actual output checks, batch/replay assertions,
supervision record and timeline evidence before continuing.

CPU analysis:

```bash
python src/check_gate0.py  # needs the sibling historical evidence directories
python src/analyze_overlap.py --root /path/to/resource_overlap_v1
python src/summarize_baseline.py
python src/make_baseline_figures.py
```

Raw JSON, commands and logs are retained, including failures. Model weights,
credentials, CUDA binaries, full NSYS reports and SQLite traces are not published
to Git. PTX/SASS and extracted launch/interval evidence are retained for review.
