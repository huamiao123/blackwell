# methodology.md — v3 实验协议（读者应能仅凭本文复现）

## 0. 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5090 32 GB，cc 12.0，170 SM，L2 96 MiB，SMEM/CTA opt-in 99 KiB，64K regs/SM |
| 驱动 / CUDA | 590.44.01 / 13.1（容器内，独占） |
| 框架 | torch 2.12.1+cu130，Python 3.12 |
| 探针 | Triton（FP32/DRAM）、cuBLAS/CUTLASS（Tensor） |
| 计时 | `cudaEvent`（`enable_timing=True`），全部取中位数 |

**前置**：每轮实验前用 `nvidia-smi --query-gpu=memory.used,utilization.gpu` 确认无其他计算进程。
本机为独占容器，仍需每次记录。

## 1. 被测量的对象

主任务：`A[M,4096] @ B[4096,12288] -> C[M,12288]`
- BF16：`torch.matmul`，FP32 累加，BF16 输出
- FP4：`torch._scaled_mm`，E2M1 payload + E4M3 block-16 scale + FP32 global scale

探针（v3 重写）：

| 探针 | 实现 | 名义压力 | 代码 |
|---|---|---|---|
| FP32 FMA | Triton，1 element/thread，4 条独立累加链，循环内零访存，末尾写 checksum | FP32 流水线 / issue | `src/probe_fp32_fma.py` |
| DRAM | Triton，连续合并读写 `out = 2*x`，AI≈1 FMA/8 B | DRAM 带宽 | `src/probe_dram.py` |
| Tensor | BF16 GEMM 或原生 NVFP4 GEMM，2048³ | tensor core | `src/probe_tensor.py` |

每个探针必须提供：
1. 独立运行时的实测耗时（`measure_alone`）；
2. 实际执行的 CUDA kernel 名（`TensorProbe.identify()` / profiler）；
3. FP32 探针需给出 PTX 中 `fma.rn.f32` 的数量（`FmaProbe.verify_codegen()`）。

## 2. 并发协议（核心修正）

### 2.1 为什么必须改

v2 逐次提交，主机在把 B 压进队列前已经过去了 10–46 ms（实测），
导致两个任务 GPU 起始时刻相差 4.5–23 ms（实测）。见 `source_audit.md` A1。

### 2.2 现行协议

```python
gA = build_graph(main_task, rA)      # 预热 3 次后捕获
gB = build_graph(probe,     rB)
# 并发：两条流 + 一个 latch event
ready.record(ctrl)
sA.wait_event(ready); sB.wait_event(ready)
with stream(sA): a0.record(); gA.replay(); a1.record()
with stream(sB): b0.record(); gB.replay(); b1.record()
ctrl.wait_event(a1); ctrl.wait_event(b1); joint.record(ctrl)
```

必须记录并报告：
- `cpu_submit_ms`：主机发起两次 replay 的耗时；
- `start_skew_ms`：`|a0 - b0|`，即两条流实际开始时刻之差（同一条时间基）。

**同时测 A→B 与 B→A 两种提交顺序**，两者分别报告，不做平均掩盖。

### 2.3 串行基线

```python
with stream(same_stream):
    e0.record(); gA.replay(); gB.replay(); e1.record()
```

单流背靠背，实测。**禁止**用 `T_A_alone + T_B_alone` 代替（v2 的 A2 缺陷）。

### 2.4 单独基线

每个会话内、每个条件都重新测 `alone_A` / `alone_B`，与并发条件交错进行。
另测 **eager 基线**（逐次 launch），用于和历史数据对齐——因为 graph 与 eager 的
单位开销不同（实测 graph/eager ≈ 0.96）。

## 3. 实验类型（不可混用）

| 类型 | 定义 | 用途 |
|---|---|---|
| **fixed_work** | 探针重复次数固定，不随 M / 精度变化 | 观察"同一份辅助工作"对两种主任务的真实干扰 |
| **fixed_duration** | 每个 (主任务, k) 单独标定 rB，使 `T_B_alone ≈ k·T_A_alone` | 观察"同等时间占用"下的竞争 |

fixed_duration 的 `k` 是标称值，必须同时记录实测 `k_realised = T_B_alone/T_A_alone`。
Gate 1 实测 `k_realised` 中位数 = 0.966（标称 1.0），偏差 3.4%。

## 4. 指标定义

```
S_A = T_A_concurrent / T_A_alone       主任务减速（>1 = 被拖慢）
S_B = T_B_concurrent / T_B_alone       辅助任务减速
G   = T_serial_measured / T_joint      并发相对实测串行的收益（>1 才有收益）
```

**物理重叠**（只有时间线能给）：

```
task_span_overlap   = |[start_A,end_A] ∩ [start_B,end_B]|
kernel_overlap      = Σ 时间，其间至少一个 A kernel 与一个 B kernel 同时执行
```

**墙钟派生量**（不得再叫 overlap）：

```
SWSF (serial_work_saved_fraction) = 1 - (T_joint - max(T_A,T_B)) / min(T_A,T_B)
```
SWSF 只回答"相对串行省了多少时间"，与物理重叠无必然关系。

## 5. 统计

- 每条件 ≥3 个独立会话；会话内不把重复 kernel 当独立样本。
- 条件顺序打乱（seed 记录在 JSON）。
- 报告逐会话原始值 + 中位数 + 极差；不隐藏离群点。
- 置信区间待样本量足够时用 bootstrap（对会话重采样），**不得**用会话内重复充数。

## 6. 门禁

| Gate | 内容 | 通过标准 |
|---|---|---|
| 1 | M∈{16,128,1024} × 2 精度 × tensor 探针 × k=1 | 修正后 S_A 可复现（会话间差 <5%）；起跑偏差 <1 ms；两种提交顺序差 <10% |
| 2 | 三类探针各自标定 | 每类探针能在独立运行时形成可测量的压力，且 kernel 身份可验证 |
| 3 | 全 M × 3 探针 × 3 压力 | Gate 1、2 均通过 |
| 4 | 机制验证 | 有明确现象 + 可证伪假设 |

## 7. 复现命令

```bash
cd audit_v3
python tests/test_metrics_v3.py                       # CPU，16 项
python src/interference_v3.py --gate1 --Ms 16,128,1024 \
    --precisions bf16,fp4 --probes tensor --ks 1.0 \
    --sessions 3 --modes graph,legacy --json raw/gate1_v2.json
python src/analyze_gate1.py raw/gate1_v2.json
python src/timeline_capture.py                        # profiler 时间线
python src/make_figures_v3.py
```
