# Gate 4 研究记录

本阶段以 Qwen3-8B 的真实 Graph 投影为诊断对象。BF16 与 NVFP4 分别沿用 Gate 3 已验证的实现。第 18 层独立重放提供可重复的机制对照，完整推理的性能边界沿用 Gate 2。

| 问题 | 依据 |
| --- | --- |
| 假设和证伪条件 | [研究假设](research_hypotheses.md) |
| 已有数据与计时范围 | [证据审计](evidence_audit.md)、[计时协议](timing_protocol.md) |
| 主 CTA 资源和理论驻留 | [资源表](kernel_resource_model.md)、[Occupancy](occupancy_analysis.md) |
| 实验矩阵、探针及控制 | [实验矩阵](experiment_matrix.md)、[探针](probe_protocol.md)、[并发协议](concurrency_protocol.md) |
| 共享 GPU | [监督协议](gpu_safety_protocol.md) |
| 本阶段状态 | [阶段报告](final_report.md) |

[统一结果表](analysis/) 按源文件和计时范围列出逐层资源、探针验收、同工作量配对、Kernel 时间重叠及混合 CTA 必要条件。完整 E2 的请求、阶段和 GPU Kernel 区间分别记录。Gate 4.4 独立复验另含 BF16/NVFP4 主任务单独与并发时的四份 P1 设备采样。[完整 E2 实测](../gate4_inference/gate44_20261119.md) · [资源竞争复验](../gate4_inference/phase_recheck_20261119.md)
