# 有真实消费者的并发任务筛选

用 Gate 2/3 已保存的时间线重算候选开销。每一行只用该采集文件内的 GPU 区间并集与同次完整 CPU NVTX 窗口比较。公式假设该段时间可以全部从临界路径移除，给出量级判断。当前没有新运行，也没有宣称这些区间实际都在临界路径上。[来源与计算表](analysis/candidate_trace_budget.csv)

| 候选 | 命中的 GPU 时间 | 同次 P2 完整窗口占比 | 将这段时间全消除的理想倍数 |
| --- | ---: | ---: | ---: |
| NVFP4 Prefill embedding、RMSNorm、FP4 量化融合 | 0.154 ms | 0.0071% | 1.000071× |
| BF16 Prefill embedding、RMSNorm 融合 | 0.154 ms | 0.0038% | 1.000038× |
| NVFP4 主模型的 BF16 LM Head | 190.646 ms | 8.82% | 1.0967× |

## 暂缓 embedding 准备

Qwen3 有现成的 `inputs_embeds` 消费者，但当前安装的 vLLM `DefaultModelState.prepare_inputs_embeds` 在文本路径仍先执行 `model.embed_input_ids`，将其写入 staging buffer，之后才用 `PromptEmbedsState.apply` 覆盖。直接把预计算结果喂给该接口会重复 embedding，无法视为移出原有工作。默认 Gate 2 路径还将 embedding 和 RMSNorm 融合，NVFP4 带上了 FP4 激活量化。拆出全量 BF16 中间结果会额外处理 B=8、L=2048、H=4096 对应的 128 MiB 张量，并可能改变 Graph 路径。优先停止该候选。[Prefill GPU 源数据](../analysis/gate2_e2_nvfp4_p2_r01_kernel_catalog.csv.gz)

## 优先审查独立请求的 LM Head

NVFP4 模型仍运行 BF16 LM Head。Gate 3 已通过真实 `compute_logits`、`LogitsProcessor`、LM Head apply 调用与 GPU correlation，确认 257 次 Kernel。平均每次约 0.74 ms，耗时明显大于 embedding。该工作在真实推理中由采样器消费，已有源代码和时间线可复核。[Gate 3 LM Head 归因](../gate3_attribution/diagnostic_report.md)

可提前计算的输入只来自已经完成 Transformer forward 的独立请求批。`GPUModelRunner.sample` 在本机源码中先索引 `hidden_states[input_batch.logits_indices]`，调用 `self.model.compute_logits`，随后进行原来的 grammar mask、采样和输出更新。同一请求下一步依赖该采样结果，因此不能把该请求的下一步 QKV 当成独立并发对象。候选是对另一批已就绪的输入工作执行 forward，同时安排前一独立批的真实 logits 计算。

当前 `GPUModelRunner.execute_model_state` 只持有一份状态，第二次 `execute_model` 会覆盖它。CUDA Graph 还可能复用 hidden-state 输出缓冲区。不能直接在两个 Stream 上交替调用这个 Runner。准备方案必须让独立请求保有各自的 scheduler iteration、logits indices、稳定的输出切片、完成 Event 和采样状态。原始 `sample` 在终点消费已核验 logits，替换一次对应的 `compute_logits`，其余 grammar、RNG、token 结果与原始路径一致。需要在真实运行核对消费者关联和调用数。

先用真实原型 `compute_logits` 作为辅助工作量测独立耗时和竞争，记录真实 BF16 Kernel、输入哈希、两批诊断的输出以及完整数据移动。它可能偏带宽受限，FP32 FMA 探针数据不能作为该候选的放行条件。0.74 ms 的整次 Head 也无法完整藏进一次约 20 µs 的 QKV，需要观察跨多个真实 Kernel 的重叠窗口，或另做确实被原采样器消费的 vocab 分段对照。分段若改变 Kernel 或数值，保持为独立方案，不与原整次 Head 对照混算。

正式验证保留同一请求集合与到达方式，比较原始 `LLM`、普通并发、资源感知调度三种条件。每批 logits 只算一次并交给原消费者，同时验证请求级 TTFT/ITL/TPOT、真实输出、主任务损失与吞吐。当前只有候选的静态调用关系和原始耗时，实测并发、截止时间和推理收益均未验证。
