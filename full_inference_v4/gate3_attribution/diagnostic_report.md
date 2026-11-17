# Gate 3 首项验证：采样路径与 LM Head 归因

## 完成了什么

2026-11-17，在单独复核GPU空闲、无控制器和无残余permit后，完成一份有效的
E2 NVFP4 P2 Node补采：`gate3_e2_nvfp4_p2_r02`。首次尝试因标记器未处理收尾EMPTY调度失败，
已保留失败源码、日志和失败状态，不计入有效数据。修正经CPU测试后，在原10分钟
窗口内重试，没有延长授权，也没有自动等待空闲。

负载、配置文件Hash、Prompt Manifest、原Gate 2 worker源码Hash和GPU UUID均与原E2 NVFP4一致。
完整推理仍为Batch=8、每条输入2048、输出257；实际256次Decode、256次Graph Replay。
没有NCU、没有辅助Kernel并发、没有模型数值修改或调度配置修改。

本轮只完成Gate 3的一个归因子问题，**不代表整个Gate 3或RQ2/RQ3完成**。

## 为什么需要补采

历史日志明确显示使用V2 Model Runner。实际文件为`vllm/v1/worker/gpu/model_runner.py`，
而旧版`vllm/v1/worker/gpu_model_runner.py`并非本实验运行路径。V2在`sample_tokens → sample`
路径中调用`model.compute_logits`，后者经`LogitsProcessor`调用真实LM Head的量化方法。

旧标记只包围executor的`execute_model`。CPU审计发现，旧E2 NVFP4 Trace的2570个UNKNOWN
Kernel都有唯一Runtime Correlation，但其Launch API不在旧dispatch标记内。仅凭这个现象
和Kernel名称不能确认逻辑算子，因此新增`SAMPLE_DISPATCH`、`LOGITS`以及实际Head apply范围。
没有给逐层forward插入hook，也没有假设Python标记会随CUDA Graph Replay重放。

当前源码快照的V2文件Hash与新诊断运行中实际runner模块Hash相同。
旧Gate 2时刻全部安装源码未留同等快照，因此不把当前源码快照冒充旧时刻源码证明。
源码/config来源与Hash见[source_provenance.json](analysis/source_provenance.json)。

## 核查结果

| 核查项 | 结果 |
| --- | --- |
| 实际Kernel总数 | 134176，与Gate 2完全一致 |
| Kernel名称多重集合 | 与Gate 2完全一致 |
| 原生SM120/E2M1 Kernel调用 | 37008，不是MMA指令数 |
| Graph Replay | 256 |
| Sample调用 | 257，1次Prefill输出与256次Decode输出 |
| 旧dispatch规则下UNKNOWN | 2570 |
| 新Sample调用范围关联 | 2570 |
| 新Trace剩余阶段UNKNOWN | 0 |
| 真实Head apply关联 | 257次 |

这些归因使用**同进程、同线程NVTX范围内Launch API的唯一Correlation**关联实际GPU Kernel，
没有用CPU标记与GPU Kernel时间重叠来猜测。采样iteration还要求真实input_batch请求ID集合
与待采样调度记录严格相同，存在歧义就失败，禁止按相邻GPU时间补标签。

新Trace的“阶段UNKNOWN=0”只表示每个Kernel的Prefill/Decode调用路径已关联。
**Backbone中FFN、QKV、O、Norm等精确算子与Layer仍未全部绑定。**
也没有回写旧Trace中的UNKNOWN，旧P1条件均值的原统计边界保持不变。

## 一个已经证实的混合精度部分

实际LM Head方法为`vllm.model_executor.layers.linear.UnquantizedLinearMethod`，
权重shape为`[151936, 4096]`，dtype为
`torch.bfloat16`；257次输入均为`[8, 4096]`。
模型量化配置也将`lm_head`列入ignore。因而当前NVFP4模型执行中，LM Head仍走BF16。

由实际Head apply调用范围关联的Kernel区间并集为 **190.646 ms**，
占同次P2完整CPU NVTX窗口的 **8.82%**。
其中首次输出对应约0.778ms，其余256次Decode合计约189.868ms。
逐次实际Kernel、Launch形状、寄存器/SMEM和Correlation见
[lm_head_launches.csv](analysis/lm_head_launches.csv)。

这证明“NVFP4模型”的完整执行包含不同精度路径；它不是新的优化，也不证明该Head可以
免费并发。不能因发现它就立刻宣称一个论文创新点或把8.8%换算为可用算力。

## 调用路径统计

| 阶段 | 路径分类 | Kernel数 | 该类区间并集 ms |
| --- | --- | --- | --- |
| PREFILL | BACKBONE_AND_PREPARATION | 518 | 301.207 |
| PREFILL | LM_HEAD | 1 | 0.778 |
| PREFILL | OTHER_SAMPLE_OUTPUT_PATH | 9 | 0.046 |
| DECODE | BACKBONE_AND_PREPARATION | 131088 | 1623.107 |
| DECODE | LM_HEAD | 256 | 189.868 |
| DECODE | OTHER_SAMPLE_OUTPUT_PATH | 2304 | 7.133 |

这是P2诊断运行中的GPU活动分类。各类别时间不是P0端到端延迟分解，也不能默认彼此可相加。
`BACKBONE_AND_PREPARATION`保留模型/准备链整体；`OTHER_SAMPLE_OUTPUT_PATH`尚未进一步拆分。
没有把Kernel名字直接作为已验证逻辑算子。

新补采完整模型墙钟为2161.539ms，原P2为2164.048ms，
单次差异-0.116%。这不是加速结果或严格扰动置信区间。Graph次数、原生Kernel数与名称集合
相同，支持本次标记没有明显改变Kernel构成；每条件一次仍不足以精确估计标记开销。

## 状态与剩余工作

- **VERIFIED（限定本次新Trace）**：LM Head的实际调用、BF16方法/权重类型，以及其Kernel归属。
- **VERIFIED（限定本次新Trace）**：2570个旧规则UNKNOWN来自新标记覆盖的sample路径。
- **PARTIALLY_SUPPORTED**：标记未明显改变执行路径；仍缺独立扰动重复。
- **UNVERIFIED**：逐层FFN/QKV/O的完整Graph Node映射、动态L2/FP32 Pipe/Warp Stall/Occupancy机制。
- **UNVERIFIED**：资源可共存程度、同SM驻留、辅助计算收益、真实推理优化。

后续先处理Backbone的调用/Graph构建映射，优先复用既有Trace和源码，不重复整套P0。
不能以“Head可以量化”或“Tensor活动较低”替代资源竞争实验。

GPU工作在2026-11-17 19:26:24 CST结束，窗口已关闭，完成时无计算进程和permit残留。
见[completion_state.json](analysis/completion_state.json)。没有自动推进下一次采集或并发Gate。
原始`.nsys-rep`及SQLite留在服务器；位置与SHA见
[raw_trace_manifest.json](analysis/raw_trace_manifest.json)。原始Gate 2结果保持原样。
