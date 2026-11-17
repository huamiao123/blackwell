# Gate 3: sampling / LM-head attribution diagnostic

This stage preserves the Gate-2 workload, scripts and data. CPU inspection found
that the actual runtime uses **V2 Model Runner** (`vllm.v1.worker.gpu.model_runner`),
not the legacy `gpu_model_runner.py`. V2 calls `compute_logits` in `sample`,
which is reached from `sample_tokens`, outside the old executor dispatch range.
All 2,570 UNKNOWN kernels in the inspected E2 NVFP4 trace have one unique runtime
correlation but no containing old dispatch range. Source evidence and launch
counts support a hypothesis, not an operator assignment by kernel name.

One reviewed diagnostic: E2 NVFP4, same complete 2048+257-token Graph workload,
same input/config hash; one independent P2 Node capture. No NCU, no concurrent
auxiliary task, no model/scheduler/numerical change, no layer-forward hooks.
The new wrapper adds:

- `SAMPLE_DISPATCH|iteration|stage`, validated against actual sampling request IDs.
- `LOGITS|iteration|stage` around the actual model method.
- `LOGICAL_OP|LM_HEAD|iteration|stage` around the actual LM-head quant method apply.

The parent supervisor's idle/controller checks, finite timeout, UUID check,
ownership monitoring and own-process cleanup are retained. A separate Gate-3
window accepts only E2 NVFP4/P2 and expires within 20 minutes. No idle polling.
Data use new `gate3_` run IDs in the existing raw/logs/sqlite storage; no old run
is overwritten. A supplementary diagnostic never becomes a new P0 baseline.

Before accepting attribution, verify actual GPU launch correlations, marker
coverage, complete workload, replay count, native kernel count, runtime class,
head dtype/method, and fusion/kernel-composition consistency versus Gate 2.
Per-layer Graph nodes remain unassigned unless further build/replay evidence
supports them; this wrapper does not claim Python forward markers replay.

CPU validation:

```bash
python -m unittest discover -s full_inference_v4/gate3_attribution/tests -v
```

One accepted diagnostic is now complete: `gate3_e2_nvfp4_p2_r02`. See
[diagnostic_report.md](diagnostic_report.md) and
[analysis/diagnostic_validation.json](analysis/diagnostic_validation.json).
It verified LM-head attribution and the sampling path while retaining exactly
the same kernel-name multiset as Gate 2. The failed first run remains recorded.
The rest of Gate 3, concurrency and useful work remain incomplete. GPU work ended
at 2026-11-17 19:26:24 CST; its window is closed. No next Gate is automatically launched.
