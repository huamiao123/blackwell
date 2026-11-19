# 历史证据审计

| 证据 | 保留的用途 | 适用范围 |
| --- | --- | --- |
| Gate 2 P0 | E1/E2 完整模型执行的延迟与 TPOT 基线 | 模型基线与原 Prompt 协议绑定 |
| Gate 2 GPU Metrics | 发现设备级和阶段级活动差别 | 设备采样，寄存器驻留与 L2 命中未由此测得 |
| Gate 3 Graph binding | 真实投影节点、权重引用、对应 Grid/Block | 各精度 144 个模块及 36,864 次已匹配 Decode GEMM |
| Gate 4 单节点诊断 | 输入就绪时的投影 GEMM 与探针竞争 | 第 18 层、固定合成 activation、真实 checkpoint 权重，权重反复使用 |
| Audit v3 | 等工作量、实测串行、GPU 实际边界、缓存公平性 | 已否决的旧 Gate 9 统计不进入新结果 |

原文件及 SHA256 见 [历史来源清单](../gate4_spatial/analysis/input_audit.json)。正式模型 [Gate 2](../gate2_report.md)、[Gate 3 配对核验](../gate3_backbone/paired_report.md) 和 [旧并发审计](../../audit_v3/corrected_results.md) 保留原统计口径。
