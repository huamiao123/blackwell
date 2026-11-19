# 空间并发阶段报告

2026-11-19 的优先矩阵已覆盖 BF16/NVFP4 QKV 和 Gate/Up，证据来自独立无 Profiling 配对和 CUDA 节点轨迹。固定工作量下，Gate/Up 的 FP32 探针与 QKV 呈现相反的联合完成收益。QKV 的 BF16 Tensor Light 组合在 NVFP4 上平均损失约 1.2%–1.3%，P2 两个顺序均在主投影窗口内完成。逐次数据、置信区间、有效辅助工作量和失败记录见 [实验报告](qkv_replay_validation/qkv_gateup_20261119.md)。

## 首轮 CPU 审查存档


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


已完成的具体内容见 [实现报告](implementation_report.md)。历史CSV自动选择BF16/NVFP4的QKV、Gate/Up；四种Probe已完成SM120离线编译与静态指令核验；15项空间实验CPU测试通过。

| 最终问题 | 当前回答 | 证据状态 |
| --- | --- | --- |
| NVFP4 QKV是否更容易并行 | 未运行本轮A/B并发，不能判断 | UNVERIFIED |
| Gate/Up是否不同 | 选择与计划已冻结，没有并发观测 | UNVERIFIED |
| Grid与完成量是否相关 | 实现等总工作量Grid-stride协议，关系未测 | GPU结论UNVERIFIED |
| FP32与BF16 MMA竞争是否不同 | FFMA/HMMA.BF16静态路径确认，竞争未测 | 指令VERIFIED（离线），竞争UNVERIFIED |
| 主任务性能损失 | 无P0观测 | UNAVAILABLE |
| 辅助任务完成量 | 逻辑工作量可核算，GPU输出/完成量未验收 | GPU结论UNVERIFIED |
| 是否支持空间并发假设 | 小Grid是动机，不足以确认H1/H2/H3 | UNVERIFIED |
| 同SM是否共驻 | 没有主CTA级直接位置证据 | SAME_SM_CO_RESIDENCY_UNVERIFIED |

没有用相近GEMM替代真实主Kernel，没有将编译成功或CPU测试当作并发成功。独立诊断Graph仍缺活激活缓冲区和输出oracle适配的验收；若不能解决只能退回阶段级，并明确改变问题范围。

`concurrency_runs.csv` 等暂为只有表头的结构，不绘制空白数据对应的性能结论图。下一步是人工审查后决定是否授权有限的GPU正确性试点，不自动扩展实验。
