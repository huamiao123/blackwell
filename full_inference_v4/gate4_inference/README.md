# Gate 4.4 完整推理并发

RTX 5090 上的完整 E2 实验采用 Qwen3-8B BF16 与原生 NVFP4，分别验证 Prefill、Decode 和生命周期并发。请求延迟和 GPU 重叠分别由无 Profiling 配对和独立 CUDA Node Timeline 测量。[实测报告](gate44_20261119.md)

数据在 `analysis/gate44_20261119/`，运行入口为 `src/run_gate44.py`。每次运行保留压缩结果、监督信息和原始 Trace 哈希。有真实下游消费者的性能实验归入 Gate 4.5。

## 初始 CPU 准备档案


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


本任务依赖Spatial/Residency的有效GPU结果，目前前置结果尚未取得。因此只实现计时/注入/真实消费者/默认回退策略契约，不选择未经证实的有用任务，不运行推理，不声称调度优化。

- [阶段报告](final_report.md)、[方法](methodology.md)。
- [Prefill协议](prefill_protocol.md)、[Decode协议](decode_protocol.md)、[生命周期协议](lifecycle_protocol.md)。
- [有用工作定义](useful_work_definition.md)、[策略设计与限制](scheduler_design.md)。
- [历史P0基线](analysis/historical_p0_baseline.csv)、[6项CPU测试](analysis/cpu_test_results.json)。

完整推理基线、历史合成输入性质及测量范围不变。新并发结果表只有表头；不能将原P0数字填入新A+B观测。
