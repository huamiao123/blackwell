# Gate 4三任务：CPU侧审查入口


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


| 任务 | 报告 | CPU结果 | 当前主要缺口 |
| --- | --- | --- | --- |
| 真实投影空间并发 | [spatial](gate4_spatial/final_report.md) | 15项测试、4种Probe离线SASS | 活激活输入/输出适配与GPU等价性 |
| CTA资源驻留 | [residency](gate4_residency/final_report.md) | 4项测试、56行理论模型 | 活函数/API、实际混合驻留与CTA直接证据 |
| 完整推理有用并发 | [inference](gate4_inference/final_report.md) | 6项测试、默认回退 | 前置实测、有用候选及实际消费者、E2E收益 |

共25项CPU测试通过，不代表GPU通过。权重未加载，GPU实验未启动，当前窗口未启用。细节与下一步审查事项见 [实现报告](gate4_spatial/implementation_report.md)。

最值得验证的机制区别：NVFP4的小Grid可能留下其他SM执行机会，但其大CTA未必允许辅助CTA在同SM驻留；两者不能混为一谈。CPU粒度模型提示FP32 Light即使全局寄存器加法通过，也可能被分区约束排除。此处仍是待GPU检验的模型预测。
