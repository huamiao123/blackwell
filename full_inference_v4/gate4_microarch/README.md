# RTX 5090 projection counters

[中文实验报告](final_report.md) · [投影资源与历史并发对照](analysis/projection_comparison.csv) · [指令地址采样](analysis/pc_waits.csv)

## Scope

One foreground job profiles the third replay of one private, single-node projection graph. The worker reuses the Gate 3 checkpoint binding, live kernel checks and bitwise output oracle in `gate4_spatial/src/validate_qkv_replay.py`. Two warmups, an explicit profiler range, one profiled action, unchanged clocks and kernel replay are recorded. `cache-control=none` preserves the profiler's existing cache state. The `all` contrast requests a cache flush at each replay pass. Both policies include profiler intervention.

## Collection and replay

On the experiment host the project must already contain the Gate 2 supervisor, E2 configurations, Gate 3 selections, checkpoint files, vLLM environment, Nsight Compute 2026.2.1 and the authorized `gpu-profile` wrapper. Deploy the tracked `full_inference_v4` source changes to the corresponding project paths. A single invocation from the client submits one scoped, time-limited measurement and fetches its evidence. The shared supervisor checks other GPU users before and during execution. A failed run returns an error and creates a case record.

```text
python src/submit.py <ssh-alias> nvfp4 down_proj resources
python src/submit.py <ssh-alias> bf16 gate_up_proj resources --cache all
python src/submit.py <ssh-alias> nvfp4 gate_up_proj pc
```

The counter lists are frozen in [counters.json](configs/counters.json). The offline chip catalog was queried with `ncu --query-metrics --chips gb202 --query-metrics-mode all`. The PC counters come from the installed SourceCounters section. The narrow L2 set uses read-hit and read-miss sector counts. Its actual pass count is checked after collection.

Run the analysis from this directory with the history from commit `db46e14` available in the parent project.

```text
python src/analyze.py
python src/window_bounds.py
python src/compare.py
python src/summarize_special.py
python src/figure.py
python src/full_dram.py
python src/figure_decode.py
python -m unittest discover -s tests
```

Each `raw/<run_id>/` contains the workload, sanitized command, exact model/kernel identity, import of the NCU report, mangled function name, and a SHA256 manifest of the original report on the experiment host. PC runs also retain the sample counts, instruction at the sampled PC, and the immediately preceding static instruction. `src/sass_inventory.py` and `src/telemetry.py` run on saved reports and monitoring files on the host. Native NCU reports remain on the host. P3 reports include 2-second supervisor telemetry and independent hardware counter cycles; thermal and power readings sampled over process setup have a different scope than an individual projection.

In [summary.csv](analysis/summary.csv) statistics are stratified by precision, operator, counter set and cache policy. Out-of-range hit probabilities retain the original value in [measurements.csv](analysis/measurements.csv), with a quality flag. `processes` counts all measurements and `valid` counts included values. The duration comparison uses empirical percentiles across nodes from one separate full-inference P2 trace per precision. Archived P0 performance statistics use paired blocks.
