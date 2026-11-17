# 同SM共驻的证据协议

| 层次 | 当前能力 | 允许结论 |
| --- | --- | --- |
| 理论必要条件/分区模型 | CPU已实现 | 特定资源假设下可行/不通过，不代表实测 |
| 单函数Occupancy API | 接口设计，未调用 | 未来仅是同类函数理论容量 |
| GPU Kernel区间重叠 | 分析器CPU测试通过，暂无真实采集 | 未来可标CONCURRENT_KERNEL_EXECUTION_VERIFIED |
| 主CTA与辅助CTA直接同SM | 未实现可靠主CTA定位器 | SAME_SM_CO_RESIDENCY_UNVERIFIED |

只取得辅助Probe的SM ID无法知道主CTA同时位于哪里。只有Grid、时间重叠、低SM Active、Occupancy或布尔声明不构成直接共驻证据。当前`evidence.py`故意不提供将此类声明升级为直接证明的路径。

若资源模型在固定寄存器/SMEM假设下明确不允许同时驻留，而Kernel执行区间又重叠，可将不同SM执行或CTA时间交错作为候选解释，标PARTIALLY_SUPPORTED；Kernel区间包含CTA等待与执行，不能仅凭此排除更细粒度交错或动态行为。

首轮QKV Light/Heavy与Gate 4 Spatial共用同一A/B工作定义、实际函数和时间线。先做QKV四个精度/资源主对照，之后再决定Gate/Up。不得因该任务理论表生成而自动启动GPU或自动推进完整推理优化。
