# 阶段注入、压力等级与公平共存实验设计

**状态：设计完成；辅助 Kernel GPU 正确性、实际注入、并发和收益均 UNVERIFIED。**
本轮不会运行，现有 worker 也没有自动注入辅助任务。

## 压力与工作量

先验收计算结果、实际编译指令、数据类型、完整独立工作量以及至少5次独立性能。
源码中的 FP32/MMA 名称不能替代实际路径。DRAM 的独立输入/输出分别 256 MiB，
均超过历史96 MiB L2；仍需访问模式与实际带宽证据。

建议初始有限方案由 `validate_probe.pressure_plan()` 输出：

- FP32：固定8192逻辑行、4条FMA链、每行8000迭代，128线程块，CTA=4/16/64。
  需要 grid-stride 分配固定行任务。各等级操作数与总FLOPs相同。
- DRAM：固定256 MiB读、256 MiB写、128完整passes，CTA=64/256/1024，块形状固定。
- Tensor：固定64逻辑 tile任务，CTA=4/16/64，使用persistent/grid-stride分配；
  GEMM尺寸、内部重复次数与资源配置在验收后冻结。

这些数字是预先披露的候选，不是已找到的最佳配置。所有调整与失败都记录。
旧 FP32 Probe 直接增加 `ctas` 会增加总工作量，不能直接用于此固定工作量压力扫描。
增加迭代次数改变持续时间/工作量；增加 CTA 只改变潜在瞬时并行度，并不证明实际共驻。

## P/D/L/M 注入

A 始终是完整真实推理，不裁成孤立算子或 Decode-only 来报告正式结果。

| 条件 | 提交锚点设计 | 实际有效性验收 |
|---|---|---|
| P | 首个真实 Prefill GPU 提交附近，以独立流等待正确的 ready Event | B 首/末 Kernel 与实际 Prefill 活动区间相交；记录是否溢出到 Decode |
| D | 首个已识别 Decode 的实际提交处 | 由调度身份、Graph Replay 和 GPU 时间线确认，不用首次输出 CPU 时间代替 |
| L | 完整请求开始附近一次提交固定有限 B | 报告 Prefill/Decode 各自覆盖、A内已完成工作及尾部 |
| M | 存在实际 MIXED 调度时选定提交点 | 区分同 Iteration 混合与相邻阶段在 GPU 上重叠 |

ready Event 仅表示不早于某点，不保证同时启动。记录 CPU提交、GPU第一/最后节点、
实际交集与起跑偏差。B跨阶段时按真实覆盖报告，不强行称纯P/D。不能为了限定阶段
而插入全设备同步或假设可立即停止正在运行的辅助 CTA。

## 同工作量配对

每个Block包含 A-alone、B-alone、实测serial AB、实测serial BA、parallel AB、parallel BA、
空对照。顺序预先平衡或固定seed随机化。同一B等级在BF16/NVFP4间保持完全相同。
同一Block内重新测A/B参照，不复用很早的一次校准。

分别报告 IA=Aparallel/Aalone−1，RB=Throughput_Bparallel/Throughput_Balone（固定工作量时
等价 Bsolo_span/Bparallel_span），S_AB=serialAB_joint/parallel_joint，S_BA同理。
另报告使用较快真实串行顺序的保守S，不择优混合分子分母。

A结束时只等A自己的结束Event，不把B尾部算进A。Joint单独等待两者。B在A结束前
只算完整确认完成的chunk；部分Kernel不按时间比例折算工作量。若持续辅助模式
工作量取决于A速度，需单独统计，不能混入固定工作量实验。

平均5%阈值之外，还检查逐请求TTFT/ITL/P95、Prefill、Decode以及完整E2E。较长Prefill
不能掩盖Decode降速。Kernel交集不证明同SM共驻；寄存器/SMEM相容不证明硬件实际调度。

## Teacher Forcing

`run_teacher_forcing.py` 只生成并验证对照计划，明确拒绝 `--execute`。
参考续写以Prompt Hash索引，保存257个输出、前256个后续输入及Hash。未来需在正常
前向后控制下一步实际输入，保持一次Prefill+256次cached one-token Decode；不能将
所有历史Token拼成一个大M Prefill替代。vLLM sampler接入、Graph兼容性与强制ID实际
消费尚未实现或验证，两精度必须承担相同接入开销。它不替代真实自回归性能。

## 有用计算

Gate 6通过后才选择真实消费者。需要输入已就绪、后续实际使用、无重复无效工作，
对照原串行、普通并发、新资源感知策略的P0端到端收益。人工FMA完成量仅是共存证据，
不能称为LLM加速。若只有负结果，保留适用负载及资源约束范围。
