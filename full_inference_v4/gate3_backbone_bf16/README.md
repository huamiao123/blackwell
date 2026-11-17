# Gate 3 matched BF16 control

One E2 BF16 full-inference P2 capture uses the same graph-parameter inspection
protocol as `../gate3_backbone/`. It targets actual BF16 WMMA kernel nodes and
binds them to real projection tensor addresses. Original NVFP4/Gate-2 sources
and data remain unchanged. No graph modification, NCU, or concurrent work.

`bf16_observer.py` reuses the frozen collector's lifetime/driver-size checks;
its node filter and `matrix_kernels` field explicitly describe BF16. The BF16
entry point records both its own hash and the observer hash in addition to the
shared wrapper hashes. The supervisor has a separate, BF16-only window check.

The accepted control `gate3_bb_e2_bf16_p2_r02` passed exact weight-reference and
CUDA tools-ID checks for 144 modules and 36,864 Decode GEMM executions. Its kernel
composition matches the original Gate-2 BF16 trace. See
[paired_report.md](../gate3_backbone/paired_report.md).
The r01 collector schema failure is preserved and excluded. One diagnostic per
precision does not establish population performance statistics or useful concurrency.
The original deadline was retained on retry; the GPU window is now closed.
