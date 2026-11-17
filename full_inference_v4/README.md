# Full inference V4.1 — Gate 2 E1/E2 results

The authorized Gate-2 GPU work is complete. Start with
[gate2_report.md](gate2_report.md) for measured results and limitations.

- P0: 40 full inference generations, four conditions, two independent processes
  with five generations each. Legal 2048-token inputs, 257 outputs, 256 verified
  Decode iterations, distinct batch prompts, prefix caching disabled, CUDA Graph.
- P1: four independent GPU Metrics captures. P2: four independent node traces.
- Local offline preencoded `LLM.generate` window; not online service E2E.
- RQ1 performance/path evidence obtained; single-capture resource observations.
  RQ2/RQ3 remain UNVERIFIED. No auxiliary concurrency or NCU experiment ran.
- GPU work ended at 2026-11-17 18:37:42 CST. The window is closed; no experiment
  process/permit remained at completion. No next gate is automatically launched.

Original CPU preparation remains documented in
[implementation_report.md](implementation_report.md),
[history_audit.md](history_audit.md), [methodology.md](methodology.md),
[prompt_protocol.md](prompt_protocol.md), [timing_protocol.md](timing_protocol.md),
[metric_inventory.md](metric_inventory.md) and
[concurrency_protocol.md](concurrency_protocol.md).
The original `analysis/artifact_manifest.json` and `quality_validation.json`
describe the CPU preparation snapshot at commit 979d88d, not the later GPU run.
The separate [Gate-2 validation](analysis/gate2_quality_validation.json) and
`analysis/gate2_artifact_manifest.json` cover this release.

The 88 tokenizer-validated inputs and E1–E7 configurations remain available.
Only E1/E2 ran. Natural-text generalization, teacher forcing, dynamic requests,
operator attribution, interference and useful work remain future reviewed gates.

CPU-only reproduction of the published derivations (repository root):

```bash
python -m unittest discover -s full_inference_v4/tests -v
python full_inference_v4/tools/test_resource_aggregation.py
python full_inference_v4/tools/derive_gate2.py
python full_inference_v4/tools/validate_gate2.py
python full_inference_v4/tools/plot_gate2.py
python full_inference_v4/tools/report_gate2.py
```

Original `.nsys-rep`/SQLite files stay on the server. Their paths and hashes are
in `analysis/gate2_raw_trace_manifest.json`. Lossless gzip copies retain raw
metric samples, kernel intervals and launch metadata. CPU export/validation
tools are in `tools/`; they require server-side SQLite files but launch no GPU.
Measured worker `src/` files are frozen to the hashes in every accepted raw run.
Future model runs still require a bounded authorization window, idle GPU and
explicit `--execute`; there is no idle polling, background watcher or gate queue.

Local artifacts: `D:\Apaper\北交\test\full_inference_v4\`.
Server artifacts: `~/fp4-llm/results/full_inference_v4/`.
