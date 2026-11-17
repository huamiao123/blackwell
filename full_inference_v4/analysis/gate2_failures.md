# Gate 2 runtime integration record

- `gate2_e1_bf16_p0_r01`: model loaded and requests executed, but the tracking
  framework failed at final response handling with external request ID `2`.
  The vLLM input processor appends a random suffix to internal IDs; the output
  processor restores the external ID in RequestOutput. The prepared worker
  tracked only the internal ID and incorrectly used the external ID as a key
  at completion. No performance sample was accepted. Full supervisor/log retained.
- Fix: explicit external→internal mapping using add_request arguments and return
  value. EngineCore schedules/outputs stay keyed by internal ID; completion and
  final public outputs resolve through the mapping. No ID-prefix guessing and
  no disabling vLLM ID randomization. CPU fake-engine regression now returns
  an external ID at completion. Later failures save their raw tracking state.

This changes instrumentation only, not model arithmetic, workload or scheduling
configuration. New valid P0/P1/P2 comparisons must all use the corrected source
hashes; the failed first attempt is not a baseline.

- `gate2_e1_bf16_p0_r02`: precheck at 14:22 rejected an existing foreign
  compute process. No new workload was launched; no foreign process was changed.
- `gate2_e2_bf16_p2_r01`: precheck at 18:04 rejected an automatic controller.
  A one-shot inspection at 18:07 found neither controller nor compute workload.
  The retry used the original authorization deadline, not an extended window.
- One CPU-only trace aggregation exceeded its shell timeout. The remaining
  Python process was identified by its open project SQLite descriptor and stopped.
  Original traces were unchanged. A linear interval scan replaced repeated union
  sorting in postprocessing; 100 randomized CPU differential cases exactly match
  the frozen reference implementation. No GPU capture was repeated for this.

Failed/rejected invocations remain in raw/logs and are excluded from P0 statistics.
All 40 accepted P0 generations and all 8 accepted P1/P2 captures use identical
corrected worker sources, matching configurations and the same prompt manifest.
