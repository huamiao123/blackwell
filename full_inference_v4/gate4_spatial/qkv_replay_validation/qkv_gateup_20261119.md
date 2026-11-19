# RTX 5090 投影并发实测

**日期** 2026-11-19，Asia/Shanghai。

同一个 64 CTA FP32 Light 探针在 NVFP4 QKV 上取得联合完成收益，在 NVFP4 Gate/Up 上却使联合完成时间超过实测串行的两倍。NVFP4 QKV 与 64 CTA BF16 Tensor Light 的结果通过六块无 Profiling 复测，主任务平均损失约 1.2%–1.3%，辅助吞吐保留约 89%，联合执行相对真实串行为 1.65–1.68×。

## 工作负载和计时边界

重放 Qwen3-8B 第 18 层 Gate 3 中实际选用的 CUDA Graph 投影。两种精度使用同一合成 activation，分别使用对应 checkpoint 的真实权重。NVFP4 输入已按原路径预量化，回放范围为核心 GEMM。Gate/Up 和 QKV 各自核验真实函数、Graph Node 参数引用、Grid、Block 和函数资源属性。每次运行先要求三次逐元素完全重放，再检查每个条件的主投影及探针结果。权重反复用于单层诊断，完整 vLLM 工作集及缓存状态需单测。

- FP32 Light 和 Heavy 每次均完成 65,536 个逻辑 item，每 item 有 16 个 FP32 值，各做 8 轮 FMA。每个有效条件为 16,777,216 FLOP。Heavy CTA 的线程数、共享数据组织和独立执行时间同时变化，属于一组可解释的资源配置对照。
- Tensor Light 每次完成 256 个 16×16 输出 tile，每 tile 执行 128 轮 K=16 的 BF16 MMA，逻辑工作为 268,435,456 FLOP。两种精度的 Probe 直接对比保持完全相同工作。
- P0 每组六个配对块，两种精度交替，独立进程，A/B 独立基线、AB/BA 实测串行及并发、空提交、同 Stream 和显式依赖均执行。性能统计采用 CUDA Event 包络。
- P2 每组合每种精度一份独立 trace，用 CUDA 实际 Kernel Start/End 做交集。P2 与 P0 分开。只有 B 的全部 Kernel 区间位于 A 内时，才能确认相应 B 工作在该观察窗口里全部完成。

## P0 六块配对结果

(I_A) 为主投影 Event 包络变化，(R_B) 为相对独立执行的辅助吞吐，(S) 为同工作量同提交顺序的实测串行联合时间除以并发联合时间。表中显示 AB、BA 的均值范围。显著性判断使用原表中的六块配对 t 区间。

| A + B | 精度 | (I_A) | (R_B) | (S) |
| --- | --- | ---: | ---: | ---: |
| QKV + FP32 Light 64 | NVFP4 | +4.33% 至 +7.58% | 89.62%–89.72% | 1.33–1.34× |
| QKV + FP32 Light 64 | BF16 | −0.45% 至 +0.18% | 86.77%–88.71% | 1.38–1.39× |
| QKV + FP32 Light 128 | NVFP4 | +9.77% 至 +10.48% | 76.22%–76.74% | 1.48× |
| QKV + FP32 Light 128 | BF16 | +5.12% 至 +5.54% | 75.97%–76.81% | 1.55–1.68× |
| Gate/Up + FP32 Light 64 | NVFP4 | −0.05% 至 +0.25% | 12.85%–12.86% | 0.412–0.416× |
| Gate/Up + FP32 Light 64 | BF16 | +4.54% 至 +4.90% | 23.01%–23.26% | 1.02–1.10× |
| QKV + FP32 Heavy 64 | NVFP4 | +16.41% 至 +17.02% | 73.01%–74.39% | 1.38–1.39× |
| QKV + FP32 Heavy 64 | BF16 | +34.97% 至 +36.98% | 92.07%–92.51% | 1.33–1.35× |
| QKV + BF16 Tensor Light 64 | NVFP4 | +1.16% 至 +1.28% | 89.21%–89.35% | 1.65–1.68× |
| QKV + BF16 Tensor Light 64 | BF16 | +0.83% 至 +3.43% | 82.34%–82.65% | 1.59–1.64× |

64 对 128 CTA 的同进程比较显示，NVFP4 QKV 的联合时间延长 15.78 和 16.03 μs。AB 干扰变化的区间跨零，BA 干扰减少 6.14 个百分点，95% 区间为减少 0.89–11.39 个百分点。64 CTA 的 NVFP4 主任务损失区间上界仍超过预设的 5%。[64/128 P0 全量统计](analysis/qkv_grid64_p0_statistics.csv)

Gate/Up 的 NVFP4 无 Profiling (S) 的 95% 区间为 AB 0.401–0.430×、BA 0.407–0.417×。探针的 P2 轨迹约在 A 启动后 46 μs 才进入，约 21 μs 的 GPU 执行区间重叠之后，仍有 204–259 μs 的 B 尾部。只检查是否重叠或主任务干扰，会漏掉这类退化。[Gate/Up P0](analysis/gate_up_grid64_p0_statistics.csv) [Gate/Up P2](analysis/gate_up_grid64_trace/qkv_pilot_summary.csv)

FP32 Heavy 的联合时间在两种精度上均低于实测串行，主任务损失超过 5% 预算。NVFP4 承受的主任务相对干扰比 BF16 低 18.56–19.96 个百分点，而辅助任务的吞吐保持率也低 17.68–19.50 个百分点。固定总 FP32 FLOP 和 Grid 仍不足以预判两边如何分担损失。[Heavy P0 全量统计](analysis/fp32_heavy_qkv_grid64_fixed_p0_statistics.csv)

QKV 的 Tensor Light 对照中，NVFP4 相对 BF16 的 B 吞吐保持率增加 7.01 个百分点（AB，95% 区间 +3.63 至 +10.39）和 6.55 个百分点（BA，+4.42 至 +8.68）。NVFP4 两种顺序的 (I_A) 区间上界为 2.37% 与 3.54%。联合收益的跨精度差值区间跨零。该结果支持单投影对潜在共存区间的区分，联合收益的精度优势仍不确定。[Tensor Light P0](analysis/tensor_light_qkv_grid64_fixed_p0_statistics.csv)

## P2 的重叠和尾部

| 64 CTA 探针 | 主投影 | NVFP4 B 尾部 | BF16 B 尾部 |
| --- | --- | ---: | ---: |
| FP32 Light | QKV | 17.31–17.44 μs | 16.80–17.41 μs |
| FP32 Light | Gate/Up | 203.55–258.98 μs | 30.98–33.57 μs |
| FP32 Heavy | QKV | 4.13–4.35 μs | 0 μs |
| BF16 Tensor Light | QKV | 0 μs | 0–0.22 μs |

Tensor Light 的两个 NVFP4 P2 条件中，B 先于 A 结束且没有 B 的前置区间，因此 B 的 256 个输出 tile 全部在主投影活动窗口内完成。其他只有部分重叠的情况无法由时长比例换算完成量。[P2 AB 时间线](figures/projection_probe_64cta_timeline.png) [QKV 的 Grid 尾部](figures/qkv_grid32_64_128_tail.png)

## 驻留解释和证据等级

Gate/Up 的 CUDA Graph 中 NVFP4 为 192 CTA、BF16 为 1536 CTA。QKV 分别是 48 和 384 CTA。实测两个投影的单一函数 Occupancy 上限分别为 NVFP4 1 CTA/SM、BF16 5 CTA/SM。一个 NVFP4 QKV CTA 与一个 FP32 Heavy CTA 仅动态共享内存就合计 120,832 B，超过 RTX 5090 的 102,400 B/SM。两者仍有 21.44–21.50 μs 的 GPU 时间重叠。该资源必要条件和时间线共同支持跨 SM 执行的解释，CTA 位置未直接观测。[资源与实测重叠表](../../gate4_residency/analysis/live_overlap_evidence_20261119.csv)

NVFP4 的小 Grid 给出空间并行的候选窗口，实际干扰由主投影和探针共同决定。不能从时间线识别同 SM 共驻、寄存器子分区调度或 Warp Stall。当前最有证据的候选是有明确依赖和截止时间约束的短 Tensor 型工作，正式请求收益仍为 `UNVERIFIED`。

## 运行完整性

10:52 的首次 64/128 P0 因 ToDesk 进入计算进程列表而停止。10:56 外部 Python 计算 Context 出现，第二次在第五块停下。保留在干净窗口完成的前四个配对，外部进程退出后完整重测第五、六块。正式清单保留被排除运行和中断证据。P0 每次启动检查计算进程、控制器和设备 UUID，进程遥测间隔无法排除所有亚毫秒级桌面活动。[中断清单](analysis/p0_grid_set_20261119_025344_utc.json) [续测清单](analysis/p0_grid_set_20261119_025727_utc.json)
