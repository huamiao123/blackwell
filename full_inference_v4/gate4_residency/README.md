# Gate 4 Residency：CTA驻留与同SM证据


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


本任务将历史主CTA、离线编译Probe与CUDA头文件粒度规则放入统一CPU模型，保留“理论必要条件”“API返回”“时间重叠”“直接同SM”四个不同层次。当前只有前一层的模型计算。

- [最终阶段报告](final_report.md)
- [资源模型](resource_model.md)、[Occupancy边界](occupancy_analysis.md)
- [共驻协议](co_residency_protocol.md)、[方法](methodology.md)
- [主资源表](analysis/cta_resource_table.csv)、[Probe资源表](analysis/probe_resource_table.csv)
- [56种理论组合](analysis/co_residency_matrix.csv)、[粒度来源](analysis/allocation_granularity_audit.json)

CPU测试由 `../gate4_spatial/tools/run_cpu_checks.py` 统一执行。不存在直接同SM测量成功的结果。
