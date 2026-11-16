# Resource overlap v1: preregistered protocol

Status: Gate 0 preparation. No new GPU result is implied by this document.

## Evidence inherited, and corrections

The nine required inputs are `nsys_round2/{final_report.md,baseline.csv,
analysis/metric_summary.csv,analysis/profiler_overhead.csv,
src/original/decode_round2.py,src/original/analyze_round2.py}` and
`audit_v3/{corrected_results.md,raw/gate9_M1024.json,source_audit.md}`.
Their SHA256 digests are recorded in `analysis/gate0.json`.

The previous real-model Graph experiment supports a resource-use difference,
not an admission or useful-work claim. Its identical prompts and enabled prefix
cache are changed here. The old M=1024 microbenchmarks are not real Decode.
The Gate 9 aggregate medians cannot reconstruct paired observations. Its
mechanistic explanation about interruption/refill remains a hypothesis.

Corrections carried forward:

- Measure both serial A→B and B→A in the same process; never add standalone times.
- Preserve every raw trial, order, work count, output check and timing boundary.
- Same finite B work for BF16 and NVFP4; never calibrate B separately by precision.
- No repeated small-weight surrogate for the real model's weight/cache traffic.
- An event is a dependency, not simultaneous admission. Report AB and BA dispatch.
- Physical overlap comes from intersecting CUPTI kernel intervals, not CUDA API,
  Graph envelopes, cumulative durations, or a wall-clock overlap formula.
- Physical overlap does not establish residency on the same SM.

## A: real autoregressive Decode

Qwen3-8B BF16 / NVIDIA NVFP4, KV fp8_e4m3, FLASHINFER attention, CUTLASS linear,
default async scheduler, CUDA Graph capture sizes [1,8], seed 0, temperature 0,
ignore EOS. Prefix caching is false. Each request has a different deterministic
2048-token sequence; both precisions receive identical IDs. Batch 1 uses row 0.
The full IDs and their hashes are saved, as are generated-output hashes.

The token budget is 16384 for **both** precisions, with chunked prefill disabled.
This accommodates all eight prompts before Decode; the old budget of 8192 could
stagger prefill when prefix caching is disabled. Every counted scheduler output
must contain exactly B requests and one token each, with pre-schedule computed
tokens >= prompt length. Any preemption, mixed prefill/Decode, incorrect token
count or Graph replay count invalidates the run.

Generate 257 outputs per request: one prefill output plus 256 Decode outputs.
Two 33-token generations warm the model. JIT, capture, prefill, validation and
model loading are outside measured Decode. Formal performance uses no profiler.

Timing starts immediately before the first actual Decode Graph replay. It ends
after the 256th output is processed and an event on the A submission stream is
complete. This includes subsequent CPU autoregressive scheduling, but excludes
initial first-step CPU preparation. Per-step latencies are CPU output-completion
intervals, not individual GPU kernel times. The instrumentation is identical in
alone/serial/parallel modes. Report this boundary separately from historical
round2's earlier scheduler-entry boundary; do not splice their latencies.

## B: finite resource probes

B is one independently captured CUDA Graph, replayed once per trial on a private
stream. All inputs exist before Decode. No host-fed infinite producer and no
mid-Decode device-wide synchronize are allowed. Validation occurs after joining.

| Probe | Work definition | Footprint and validation |
|---|---|---|
| FP32 | 4 independent FMA chains, 8 CTAs × 128 lanes, runtime iterations, fixed chunks | FP32 outputs compared to CPU `fmaf` recurrence; PTX FMA and compiler resources saved |
| DRAM | 256 MiB input copied to independent 256 MiB output per chunk, grid-stride coalesced access | Each buffer >96 MiB L2; full equality; logical bytes = 2×256 MiB×chunks; not measured DRAM transaction bytes |
| Tensor | BF16 M64 N128 K1024, 16×32×32 tiles, repeated MMA on each tile | CPU integer-matmul reference × repeats; PTX BF16 MMA required; compute/cache-resident probe, not DRAM test |
| Empty | One CTA writes a per-chunk stamp | Same graph submission mechanism; all stamps checked |

Defaults and every adjustment are saved in configuration. Low/medium/high vary
only runtime iterations (FP32/Tensor) or full-buffer passes (DRAM), not a hidden
grid sweep. A short pilot uses fewer A steps and B chunks and is not a formal
256-step result. Count two FLOPs per FMA. Tensor FLOPs include deliberate inner
MMA repetitions. All chunks must produce checked outputs or stamps.

## Paired measurements and boundaries

For each finite B configuration, collect A-alone, B-alone, serial AB, serial BA,
parallel AB, parallel BA, and empty controls. Rotate a declared order by repeat;
save every trial. Both models keep B buffers allocated even in A-alone trials.
Standalone B follows the same uncached prefill (one output, no Decode) outside
its measurement, with the model resident. This prelude is explicit, not a cache
flush and not a second model invocation.

At the first A Graph replay, a common event is recorded on the actual A stream.
Parallel AB submits one A replay then one finite B Graph; BA submits B then A.
All later A steps use the unchanged vLLM execution path. B never joins back into
the A stream. A completion synchronizes only the A event. Final joint completion
joins B separately. Serial BA completes B before A's first replay and resets the
A latency origin; serial AB launches B after A completion. Serial joint wall time
is measured directly from the common dispatch point through both completions.

Record CPU dispatch skew, A wall time, B CUDA-event span, B completion from common
event, joint wall time, A P50/P95, replay counts, request/output counts, checksums
and whether B completed by A completion. B event span includes GPU admission
delay after its start event. Standalone and parallel use the same definition.

Derived per-pair metrics: IA = Aparallel/Aalone−1; RB = Bsolo_span/Bparallel_span;
S = measured_serial_joint/parallel_joint. Report both serial orders; also use the
faster measured serial order as a conservative reference. Do not ratio independent
medians and call them paired results. Five repeat blocks are the statistical
unit; the 256 correlated token latencies are not 256 independent experiments.

If B spills beyond A, do not count its full work as work completed during A.
Unprofiled event completion can establish all-or-not-all completion. A separate
NSYS node trace counts fully contained B chunks; partial chunks have unknown
completed work. Do not prorate work by kernel duration.

## Timeline and interpretation gates

Use a separate short node-level NSYS trace. Identify B by unique compiled kernel
name and actual CUPTI stream, not Python stream IDs or kernel counts. Map the
trace GPU to the historical PCI/UUID. Compute interval unions and intersections.
Graph envelopes do not substitute for node intervals. Record registers, static
and dynamic shared memory, block/grid and kernel families when available.
Resource arithmetic gives necessary bounds, not proof of same-SM residency.

Primary initial acceptance: IA <=5%, no clear P95 regression, verified nonzero
work, measured S>1 with repeatability, and physical GPU overlap. These are
research thresholds, not hardware laws. Temporal gaps, spatial slack, pipeline
activity, and useful admitted work are distinct statements. Artificial arithmetic
is not real inference acceleration. Gate 6 needs an actual consumer and dependency
proof for any proposed useful work.

## Order and shared-GPU safety

Gate 0 CPU only → Gate 1 four fair baselines → Gate 2 probe correctness → Gate 3
NVFP4 B8 FP32 short pilot and timeline → Gate 4 matched BF16 → Gate 5 other
resources/B1 → Gate 6 useful work only when supported. Do not launch this entire
sequence unattended. Every invocation has an idle/controller precheck, ownership
tracking, two-second monitoring and hard timeout. A foreign compute process or
GPU error stops our process group only. No driver/clock/power/permission changes.
All new local artifacts are under D:, remote artifacts under ~/fp4-llm/.
