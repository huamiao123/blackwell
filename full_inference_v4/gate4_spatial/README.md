# Gate 4 Spatial：真实投影 Kernel 的空间并发准备


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


入口是 [CPU 实现报告](implementation_report.md) 和 [阶段报告](final_report.md)。本任务先研究 layer 18 的 QKV 与 Gate/Up，实际身份、Grid、Block 和资源从 Gate 3 CSV 自动读取，未重新构造同名 GEMM。

已实现实验契约、两类 Light/Heavy 探针、九条件提交协议、独立诊断 Graph 的节点复制组件、时间线分析器和公平统计。真实 vLLM 单节点重放的激活输入/输出/准备前缀适配器尚未完成验收；当前 CLI 保持禁止执行。阶段级回退仅是另一种测量范围，不能回答单投影的 H1。

相关资料：

- [方法与边界](methodology.md)、[实验矩阵](experiment_matrix.md)、[主 Kernel 选择](kernel_selection.md)。
- [探针协议](probe_protocol.md)、[实现限制](implementation_notes.md)。
- [离线编译清单](analysis/offline_compile/compile_manifest.json)、[SASS 与资源验收](analysis/probe_validation.csv)。
- [测试结果](analysis/cpu_test_results.json)、[历史输入审计](analysis/input_audit.json)。
- [资源驻留报告](../gate4_residency/final_report.md)、[完整推理准备](../gate4_inference/final_report.md)。

CPU 复核命令，从仓库根目录执行：

```bash
python full_inference_v4/gate4_spatial/tools/run_cpu_checks.py
python full_inference_v4/gate4_spatial/src/review_gate.py --config full_inference_v4/gate4_spatial/configs/review.json
```

这些命令不运行 CUDA。给当前入口添加 `--execute` 会拒绝运行。禁止手工改开关绕过后续人工审查、运行时输入证明或原有 GPU 监督机制。
