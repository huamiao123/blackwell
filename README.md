# blackwell — RTX 5090 (SM120) BF16 vs native NVFP4 resource study

## 最新实验：Qwen3-8B Decode / NSYS GPU Metrics（2026-11-16）

本轮使用真实 Qwen3-8B BF16/NVFP4 请求，对照 Batch=1/8、Eager/Graph，完成
8 个核心配置的 40 次无 Profiling 基线测量，以及 26 份 NSYS Trace/SQLite。
没有开展新的并发探针、Kernel 优化或 NCU 测试。

- [实验入口与复现说明](nsys_round2/README.md)
- [完整报告及证据等级](nsys_round2/final_report.md)
- [正常性能基线](nsys_round2/baseline.csv)
- 四张图：[GPU 时间线](nsys_round2/figures/01_gpu_timeline.png)、
  [DRAM Read](nsys_round2/figures/02_dram_read_timeseries.png)、
  [SM/Tensor Active](nsys_round2/figures/03_sm_tensor_timeseries.png)、
  [Decode 时间分解](nsys_round2/figures/04_decode_time_breakdown.png)

Batch=8 的正常 Graph 路径中，BF16/NVFP4 分别约为 10.96/6.88 ms/步；
独立 Graph 采样中的 DRAM Read 稳态均值分别约为 79.3%/44.0%。
两类数字来自不同测量。Eager CUDA tracing 有明显扰动；
**本轮确认资源使用差异，尚未证明有可无干扰利用的硬件余量。**

以下 `audit_v3/` 与首轮部分保留其原有实验背景和更正记录，不与本轮模型级数据混用。

> ## ⚠️ 修正公告（2026-11-14 晚）
>
> 首轮实验（本 README 下方与 `reports/M1_*`、`src/interference_*.py`）使用的并发协议
> **有致命缺陷**：两个任务的实际 GPU 起跑时刻相差 **4.5–23 ms**（实测），
> 串行基线是两次测量相加而非实测。
>
> 因此以下内容**已撤回**：
> - 所有 `S_A` / `S_B` / `G` 的绝对值
> - 所有 `overlap %`（那不是物理重叠率，只是墙钟派生量）
> - "FP32/DRAM 探针无余量"、"M≈256 交叉点"等结论
>
> **请看 [`audit_v3/`](audit_v3/) 的审计与修正版结果。** 唯一经受住协议修正的定量结论是：
> M=1024 时 BF16 主任务被 tensor 干扰只慢 4.8%，FP4 被慢 57%。
>
> 仍然有效的部分：原生 FP4 MMA 的 SASS 证据、六阶段数值验收、单任务吞吐表。
> 详见 [`audit_v3/source_audit.md`](audit_v3/source_audit.md) 与
> [`audit_v3/corrected_results.md`](audit_v3/corrected_results.md)。

Measurements of how BF16 and native NVFP4 (E2M1 payload + E4M3 block-16 scales)
differ in *hardware resource usage* on a single RTX 5090, and whether the FP4
side leaves capacity that a second task can actually use.

Everything here is measured on real hardware. Numbers marked "estimated" or
"derived" say so explicitly.

## TL;DR（以 `audit_v3/` 的修正结果为准）

| Question | Answer | Evidence |
|---|---|---|
| Does FP4 change tensor-pipe pressure? | **Yes, and asymmetrically**: with a tensor probe alongside, BF16's `S_A` falls monotonically 1.77 → 1.06 as M grows, FP4 stays flat at ≈1.55–1.66 | `audit_v3/figures/fig_gate3_crossover.png`, `audit_v3/raw/gate3_summary.csv` |
| Does FP4 change scalar/DRAM pressure? | **Not established.** The old "S_A ≈ 2.0 for both" used impure probes (`torch.add`/`torch.sum`); they were rewritten in `audit_v3/src/probe_*.py` and only the probes themselves were validated (Gate 2), not the full matrix | `audit_v3/raw/gate2.json` |
| Can a second task really run concurrently? | **Kernel intervals do overlap (24–88%), but there is no evidence of same-SM co-residency.** Even a 1-CTA probe only reaches 24% kernel overlap against a full-grid GEMM | `audit_v3/raw/timeline_overlap.json`, `audit_v3/raw/gate4.json` |
| Compute ceiling | BF16 212.6 TFLOPS, FP4 1156.0 TFLOPS (both measured, same kernel family) | `data/bench_precision_first.json`, `audit_v3/raw/gate2.json` |

Key corrected result: **the BF16/FP4 tensor-interference curves cross at M ≈ 32–64, not at
M ≈ 256** as the first report claimed. BF16 tolerates a tensor probe better and better as M
grows; FP4 does not improve.

## Layout

```
nsys_round2/ ← 2026-11-16：真实模型 Decode、NSYS 10 kHz Metrics 与执行模式对照
  final_report.md  README.md  baseline.csv
  src/original/  tools/  raw/  analysis/  logs/  figures/
audit_v3/   ← 修正版实验（审计 + 新协议 + Gate 1–4），先看这里
  source_audit.md       source audit (A1–A12, file:line, severity)
  methodology.md        reproducible protocol
  hypotheses.md         H1–H5 with support / counter-evidence / falsification tests
  corrected_results.md  verified / withdrawn / open, with Gate 1–4 numbers
  src/                  rewritten probes, CUDA-Graph concurrency, timeline analysis
  raw/  figures/        raw JSON + logs, figures
src/        首轮的测量代码（已被 audit_v3 取代，见 src/DEPRECATED.md）
tests/      数值验收套件（仍有效）
scripts/    环境快照 / 原生 MMA 证明
reports/    首轮报告（含撤回公告）
data/  figures/         首轮数据与图
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
