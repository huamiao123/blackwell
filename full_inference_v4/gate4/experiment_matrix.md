# 实验矩阵

| 主投影 | 辅助探针 | 确认方法 |
| --- | --- | --- |
| QKV | FP32 Light，32、64、128 CTA | P2 逐 Grid，P0 两组同进程配对 |
| Gate/Up | FP32 Light，64 CTA | P2 两种精度，P0 固定 Grid 复测 |
| QKV | FP32 Heavy，64 CTA | 同 Grid、同 FP32 FLOP 的 P2 资源对照 |
| QKV | BF16 Tensor Light，64 CTA | P2 检查子区间，P0 比较可重复干扰 |
| O、Down | 继承通过验收的候选 | 参照 Gate 3 精确映射，需单独验收 |
| 完整推理 | 有下游消费者的候选 | Prefill、Decode 和请求生命周期独立验证 |

实验顺序与实测状态见 [后续对照计划](../gate4_spatial/qkv_replay_validation/continuation.md) 和 [阶段报告](final_report.md)。
