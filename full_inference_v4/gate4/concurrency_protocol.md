# 并发与注入

同一个 Context 保留主投影输入、真实权重和输出，并在两个独立 CUDA Stream 启动两个经过核验的单节点 Graph。原 Gate 3 Graph 保持归档状态。使用同一准备事件，串行 AB/BA 实测先后，由 P2 实际 CUDA 节点活动确认。空提交、同 Stream、显式依赖三个对照分别排除提交成本和负对照误标。

完整推理优先在独立 vLLM 诊断入口采用阶段级提交。如果需要精确 Graph 注入，应重新验收受影响节点、依赖边、图形状与输出。真实推理优化需额外证明辅助任务被执行路径消费。 [空间实施说明](../gate4_spatial/implementation_notes.md) [完整推理调度](../gate4_inference/scheduler_design.md)
