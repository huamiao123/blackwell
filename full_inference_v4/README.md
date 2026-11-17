# Full inference V4.1 — Gate 0/1 CPU preparation

**Current scope: CPU preparation only. No model inference, CUDA workload,
NCU injection, GPU contention test, or Gate 2 auto-advance has been performed.**

Start with [implementation_report.md](implementation_report.md). The research
questions and evidence limits are in [research_hypotheses.md](research_hypotheses.md).

- [Historical evidence audit](history_audit.md)
- [Prompt/tokenizer protocol](prompt_protocol.md)
- [Timing and asynchronous attribution](timing_protocol.md)
- [Metric availability](metric_inventory.md)
- [Concurrency and pressure protocol](concurrency_protocol.md)
- [Overall methodology](methodology.md)

88 tokenizer-validated inputs cover eight distinct rows, five synthetic lengths
and six natural-text lengths. Four natural prompt formats are represented by
public-domain literary reading, PSF-licensed Python code and authored dialogue
around those sources. This small English benchmark does not represent production
traffic or broad application quality. Full source notices and hashes are retained.

E1–E7 each have matched BF16/NVFP4 configurations. Only E1/E2 can pass the future
Gate-2 supervisor scope check. The remaining configurations are preparation for
later reviewed gates, not authorization to execute them.

CPU commands (from repository root):

```bash
python -m unittest discover -s full_inference_v4/tests -v
python full_inference_v4/src/validate_prompts.py --tokenizer-dir /path/to/canonical/Qwen3-8B-BF16
python full_inference_v4/src/run_full_inference.py --config full_inference_v4/configs/E1_bf16.json
python full_inference_v4/src/run_supervised.py --config full_inference_v4/configs/E2_nvfp4.json --run-id cpu_plan
```

The last two commands are dry runs and do not query the GPU or load vLLM.
To regenerate prompts, use `src/prepare_prompts.py` with the tokenizer whose
asset SHA256 matches `analysis/tokenizer_asset_audit.json`. It uses local assets,
`USE_TORCH=0`, no automatic BOS, and offline tokenization. Regeneration is blocked
after measured raw runs exist. Historical data is never rewritten.

`configs/gpu_window.example.json` is disabled. Future execution requires a reviewed,
bounded window file, explicit `--execute`, an idle device and absence of automatic
controllers. Each invocation runs one configuration only; there is no idle polling,
experiment queue, scheduler, background watcher, or next-gate launcher.

Empty performance CSV files contain schemas only. CPU fixtures stay in ignored
`local_cache/` and are not GPU measurements. `raw/` and `figures/` have no new GPU
results. Teacher forcing, probe pressure variants, useful-work integration and
paper figures remain design-only or require future GPU validation.

Local artifacts: `D:\Apaper\北交\test\full_inference_v4\`.
Remote deployment target: `~/fp4-llm/results/full_inference_v4/`.
