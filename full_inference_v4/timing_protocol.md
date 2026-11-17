# 完整推理、异步调度与计时协议

## 测量窗口

当前准备的是 `LLM.generate` 离线、预编码输入。它包括调用期间请求准备、Engine
排队/调度、Prefill、全部 Decode 和输出处理；不包含客户端网络和本次 Tokenization。
Tokenization 已在数据准备阶段完成，不能将本结果称为在线服务 E2E。

- `full_model_wall_ms`：进入本次 generate 到完整返回的 CPU 墙钟。
- 逐请求 Engine latency：`add_request` 调用开始到该请求 finished 输出的 Engine step。
- TTFT：上述请求开始到 output processor 处理第一个非空输出事件。
- ITL：同一请求连续输出事件的间隔；只有每事件 1 Token 时才称 Token ITL。
- TPOT：`(最后输出时间−首输出时间)/(输出Token数−1)`；单 Token 为 NOT_APPLICABLE。
- 完整输出吞吐与 Decode-only 吞吐必须分别标明分子及窗口。

P0 的输出事件来自低开销 Hook，与客户端流式接收时间不同。多 Token 捆绑事件
明确标注，不人为给 Token 填相等时间并生成虚假的逐 Token ITL。

预热、模型加载、编译与 Capture 在窗口外。没有 Prefill/Decode 交界全设备同步。
generate 返回后另行 drain 并记录结束时间，它不回填为 CPU 请求延迟。

## 三种边界

1. CPU 请求/NVTX：用户态调用和响应观察。
2. Scheduler Iteration：使用调用原 schedule **之前**的 computed/prompt 长度，按每
   请求真实 scheduled tokens 分类。记录 PREFILL/DECODE/MIXED/EMPTY，不读取已被
   乐观更新的计数后再猜测阶段。
3. GPU 活动：CUPTI start/end、device/context/stream、Graph/node/correlation。

`ITERATION_DISPATCH|iteration|phase` 包围实际 execute_model 提交。Graph Replay 标记
携带该调度身份。GPU Kernel 通过关联到 CUDA API，再按同线程、包含该 API 的
dispatch 标记赋予候选阶段，**不按 Kernel 是否落在 CPU NVTX 时间里归属**。

输出请求 ID 使用安装版 add_request 的返回值，防止内部 ID 改写导致归属错误。
所有 Hook 在 finally 恢复。TraceState 与 Fake Engine 测试不等于 vLLM GPU 集成验收。

## P0/P1/P2/P3

P0 无 CUDA tracing；每配置至少 5 次完整推理，关键结果独立进程复验、精度交替。
P1 使用 Graph 粒度与 gb20x、10 kHz，采集设备级资源。P2 独立使用 Node 粒度。
二者在完整推理之前启动 capture，正常返回并 drain 后停止；不截成 Decode-only。
P3 仅准备计划，不自动生成整模型 full-set NCU 命令。

P1/P2 对照必须匹配配置文件、Manifest 和全部 worker 源码 Hash。输出长度同为257，
实际 Decode 同为256，比较相同完整窗口。P2 允许是受扰条件下的诊断，不能回填
P0 Kernel 间隙。P3 的全应用受扰时间与目标 Kernel 重放时间也分别报告。

## 时间线验收

导出后先 PRAGMA 检查 Schema，确认 SessionNs、设备 PCI/UUID、单进程/Context、
采样覆盖、Diagnostics、Graph Replay 与阶段计数。不能加猜测偏移或把 rawTimestamp
直接与 CUPTI 比较。当前相关分析只是 CPU 测试通过，实际 Schema 必须重新验收。

Graph 外包络可作为阶段活动包络，但不能作为精确 Kernel busy time。P1 缺少 Node
时完整 Kernel Coverage 为 UNAVAILABLE。P2 用实际 Kernel 区间并集；相交算子和
阶段的时长不能直接相加解释模型关键路径。

指标使用设备级样本，保留 typeId/metricId。分阶段统计是条件视图，阶段重叠时
不能当可加资源预算。时间加权使用相邻样本分段常值假设，缺口>1.5×中位采样间隔
不延长填充；末样本之后不猜尾值。原始采样、单位和有效覆盖率必须保留。

## 逻辑算子

默认 UNKNOWN。Graph Node 映射合同绑定具体 Trace SHA、Context、Graph/Node、名称、
源码与形状证据；未捕获 Kernel 可用具体 Context/correlation 身份。不能仅按名称、
执行顺序或跨 Trace 复用 Node ID。融合 Kernel 计一次，逻辑 M 与 padded/execution M
分开，未证实的执行形状保留 UNVERIFIED。
