# Gate 4 阶段报告

**记录日期** 2026-11-19

11 月 19 日下午的独立进程复验再次观察到阶段排序反转。NVFP4 与 BF16 的同一 BF16 Tensor 探针在 Prefill 中分别保留约 17.0% 和 10.7% 的阶段串行吞吐，在 Decode 中分别保留约 79.6% 和 89.6%。新增 4 份 10 kHz 设备采样、4 份 Node 时间线和 40 次无 Profiling 完整推理。Decode 探针覆盖 Graph 入口的融合算子及 GEMM。完整请求的联合完成速度仍在 1× 附近。[独立复验、资源数据与测量边界](../gate4_inference/phase_recheck_20261119.md)

## 机理阶段进展

| 问题 | 结果 | 证据等级 |
| --- | --- | --- |
| 两种精度能否与额外计算在 GPU 上重叠 | QKV 和 Gate/Up 的指定组合均观察到真实 GPU Kernel 时间重叠，分属同 Context 的独立 Stream | `VERIFIED`，仅限实际测量条件 |
| NVFP4 是否一概更容易容纳额外计算 | 一组 BF16 Tensor Light 在 NVFP4 QKV 期间保留更多辅助吞吐。Gate/Up 的 FP32 Light 反例表明不能从精度和 Grid 单独决定并发策略 | `VERIFIED`，有配对 P0 和独立 P2 |
| Grid 变化影响什么 | QKV FP32 Light 的 64 CTA 比 128 CTA 具有更长的 B 尾部。相同工作量下 NVFP4 的联合时间约多 16 μs，部分顺序主干扰降低 | `VERIFIED`，对应固定工作量范围 |
| NVFP4 重 CTA 会否限制同 SM 共驻 | 数个配置超出混合资源必要上界。QKV + FP32 Heavy 仅 Dynamic SMEM 即超限，同时观察到 GPU 区间重叠 | `PARTIALLY_SUPPORTED`，支持跨 SM 解释 |
| 是否直接验证同 SM 共驻 | 当前采集没有 CTA 级 SM 位置 | `UNVERIFIED` |
| 完整推理能否容纳额外任务 | E2 的 Prefill、Decode 和生命周期轨迹均验证固定探针重叠，串行负对照为零重叠 | `VERIFIED`，指定 E2 与短探针 |
| 能否改善真实推理性能 | E2 短探针的联合请求收益需以无 Profiling 成对结果判定；真实消费者的优化收益尚缺 Gate 4.5 实测 | `UNVERIFIED`，真实推理收益 |

[P0、P2 数值和失败记录](../gate4_spatial/qkv_replay_validation/qkv_gateup_20261119.md) · [模型与实测驻留关联](../gate4_residency/analysis/live_overlap_evidence_20261119.csv)

## 软件与数据完整性

逐层资源表由已有 Gate 3 原始表自动关联 288 条记录，记录 Graph、Node、CUDA Tools ID、各维 Grid/Block 和占用上界。CPU 侧覆盖 32 项正确性检查。六种 DRAM Streaming 变体完成 SM120 离线编译和访存指令审核，实际 DRAM 性能及正确性待 GPU 验收。历史 Gate 2/3 文件保留原内容。

场景限定于稳定输入的独立 CUDA Graph。原始源码、真实执行、PyTorch Graph 捕获与输出一致性均经过核对。全模型调度、全权重工作集和混合请求属于另一个适用范围。单机配对置信区间只描述当前设备和运行协议。

## 研究推进边界

任务一首轮的 QKV 和 Gate/Up 矩阵有四组六块 P0 配对和八份独立 P2 Trace。后续固定 64 CTA BF16 Tensor 探针，扩展到 O、Down 与 Gate/Up。NVFP4 的 O 和 Down 主 Grid 均为 32 CTA，Down 在实际轨迹中的探针完成余量更长，无 Profiling 的六个配对块支持 5% 主任务延迟预算。Gate/Up 则使联合完成时间比实测串行长约 41%–42%。任务二已有活函数资源、同质 Occupancy API 和混合资源与时间线对照，实际 CTA 位置尚无直接证据。[跨投影实验](../gate4_spatial/qkv_replay_validation/tensor_projection_extension_20261119.md)

任务三已完成固定 E2 的阶段级 Gate 4.4 试点。完整 Prefill 和 Decode 可与固定短探针重叠，且相同探针在 NVFP4 和 BF16 上的吞吐排序随阶段改变。当前 Gate 4.5 仍须核验候选的下游消费、输入准备成本、截止时间、缓存干扰和请求 P95。[完整推理实测](../gate4_inference/gate44_20261119.md) · [真实工作候选审计](../gate4_inference/candidate_audit.md)
