# Gate 3 continuation: graph-node / projection binding

Current reviewed scope: one E2 NVFP4 complete-inference P2 diagnostic. Existing
Gate-2 and first Gate-3 files remain frozen. A post-instantiate hook reads live
Graph metadata and copies native GEMM host argument bytes at sizes reported by
`cuFuncGetParamInfo`. Module tensor addresses (weights, scales, alpha) are read as
metadata only; no GPU weight values are copied. A node is assigned only if exact
argument references identify one unique module. Ambiguous/missing references
remain unverified. No assignment by timing or assumed projection order.

The observer does not alter graph nodes, enable debug retention or change
instantiation policy. Capture metadata is connected to actual graph object
identity at replay; CUDA tools IDs are subsequently checked against NSYS.

This diagnostic targets Decode Graph identity. Prefill operator assignments,
non-GEMM chains, dynamic counters, BF16 comparison, and concurrency are not
automatically established. Failure to decode references is an instrumentation
limit, not evidence that the GPU lacks usable resources.

The original supervisor retains idle/controller checks, a bounded window,
hard timeout, ownership monitoring, and own-process cleanup. No idle polling,
NCU, auxiliary work, or automatic next gate is included.

CPU test: `python -m unittest discover -s full_inference_v4/gate3_backbone/tests -v`.

Primary API references:

- https://docs.pytorch.org/docs/stable/generated/torch.cuda.graphs.CUDAGraph.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html

The E2 NVFP4 diagnostic passed: 144 projection modules were connected by exact
weight references and CUDA tools IDs to 36,864 actual Decode GEMM executions.
The BF16 control passed separately. See [paired_report.md](paired_report.md).
Prefill, complete fused chains and useful concurrency remain unverified. GPU
work is finished and both windows are closed; no next gate is automatically run.
