# blackwell — RTX 5090 (SM120) BF16 vs native NVFP4 resource study

Measurements of how BF16 and native NVFP4 (E2M1 payload + E4M3 block-16 scales)
differ in *hardware resource usage* on a single RTX 5090, and whether the FP4
side leaves capacity that a second task can actually use.

Everything here is measured on real hardware. Numbers marked "estimated" or
"derived" say so explicitly.

## TL;DR

| Question | Answer | Evidence |
|---|---|---|
| Does FP4 change scalar/DRAM pressure? | **No.** S_A ≈ 2.0 for both, at every M | `figures/fig1_slowdown_heatmap.png` (fp32/dram rows) |
| Does FP4 change tensor-pipe pressure? | **Yes, and the direction flips with M** | same figure, tensor row |
| Can a second task really run concurrently? | **Only tensor work.** FP32/DRAM overlap 3–4%, tensor 37% (max 69%) | `figures/fig2_overlap.png` |
| Compute ceiling | BF16 212.6 TFLOPS (51% of spec), FP4 1156.0 TFLOPS (69%) | `figures/fig3_pareto.png`, `data/bench_precision_first.json` |

Key scheduling result: **below M≈256 the FP4 side has more tensor headroom;
above it the BF16 side does.** BF16's tensor slack grows monotonically with M
(S_A 1.72→1.04), FP4's is U-shaped with a minimum at M=128 (S_A 1.25).

## Layout

```
src/        measurement and reference code
  nvfp4_recipe.py          NVFP4 encode/decode recipe (FP32 global scale, E4M3 block-16)
  interference_v2.py       dual-stream interference matrix (duration-matched)
  interference_full.py     same, full M sweep
  make_figures.py          all figures
  bench_precision.py       BF16 vs FP4 throughput
  overlap_probe.py         torch.profiler trace based overlap (cross-check only)
  overlap_estimate.py      wall-clock overlap estimate (authoritative)
tests/      correctness suites (CPU + GPU)
scripts/    environment / native-MMA proof
reports/    human-readable findings
figures/    PNG output
data/       raw JSON results
```

## Reproduce

```bash
# correctness (needs a CUDA GPU)
python tests/nvfp4_acceptance.py            # 6 stages, 12 checks
# interference matrix, full M sweep (~3 min on an idle 5090)
python src/interference_full.py
# figures
python src/make_figures.py
```

## What was verified, and how

* **Native FP4 MMA**: the executed kernel is
  `cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_...`;
  its own SASS contains **256 × `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`** and
  **0 HMMA**, with a negative control returning 0 (`scripts/prove_native_mma.sh`).
* **Scale layout**: our numpy `SWIZZLE_32_4_4` matches torch's `to_blocked` on
  10/10 shapes with **max_abs_diff = 0** (`tests/test_to_blocked_equiv.py`).
* **Numerics**: six staged checks, all pass; random-input relative L2 1.7e-3
  (bf16 output rounding level).

## Honest limitations

* Probes are PyTorch ops, not hand-written kernels — compare *relative* numbers only.
* Data is synthetic (random FP4 bit patterns), not real model weights.
* 3 sessions per condition, median reported; no bootstrap CI yet.
* Overlap is a wall-clock estimate; Nsight Systems timeline not yet available.
* BF16 kernel register/smem were derived from tile×stage, not measured.

## Environment

RTX 5090 32 GB (cc 12.0, 170 SM, L2 96 MiB, SMEM/CTA 99 KiB), driver 590.44.01,
CUDA 13.1, torch 2.12.1+cu130, Python 3.12. Idle, exclusive container.
