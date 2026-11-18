# 短探针的实际 GPU Kernel 时长

北京时间 2026-11-18 22:05:53–22:05:56，在同一台 RTX 5090 上用 NSYS 2026.1.3 复测四档短探针。沿用相同 CUBIN、65536 个逻辑项、输入和工作量。全部输出均通过校验，四组输出哈希与无 Profiler 运行一致。结束时监督器确认无 GPU 计算进程。本轮使用独立探针，模型权重无需加载。[运行和导出命令](raw/probe_20261118_140550_utc/trace_manifest.json) · [监督日志](raw/probe_20261118_140550_utc/supervision.json)

分析通过每次启动的 NVTX 标签、CUDA API correlation ID 和实际 CUPTI Kernel 记录做一对一匹配。SQLite 完整性检查通过。四个条件各关联一次预热和五次正式记录，共 24 次。GPU UUID、CUDA 设备映射、worker PID、Context、Stream、函数名、Grid、Block、寄存器和动态 Shared Memory 均已核对。GPU 时长由每条 Kernel 记录的 `end-start` 得到，表中中位数除去预热。[逐启动记录](analysis/qkv_probe_kernel_intervals.csv) · [汇总](analysis/qkv_probe_trace_summary.csv)

| 轮数 | CTA | 无 Profiler Event 中位数（µs） | NSYS 下 Event 中位数（µs） | NSYS Kernel 中位数（µs） | 本次 Kernel 范围（µs） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 96.128 | 102.560 | 64.993 | 64.673–65.185 |
| 8 | 128 | 55.712 | 50.816 | 16.864 | 16.704–17.345 |
| 32 | 32 | 156.064 | 158.112 | 122.754 | 121.890–122.819 |
| 32 | 128 | 52.800 | 65.248 | 31.456 | 30.976–31.617 |

同一次带 NSYS 的运行里，逐次比较 Event 包络和 GPU Kernel 区间，四组的差值中位数是 33.855–37.375 µs。Event 包络跨越多次 Python 和 CUDA API 提交，包含实际 Kernel 区间之外的时间。本次没有把差值进一步归因到某一个 API 或某一段等待。正式并发实验应保留联合完成时间与 Kernel 时间线两个测量边界。[Nsight Systems 用户手册](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

128 CTA、8 轮的 16.864 µs 与历史 NVFP4 QKV 的平均 19.5 µs 接近，可作为首个并发试点。BF16 对照使用同一探针、同一工作量，避免把精度差异与辅助负载变化混在一起。32 轮的 31.456 µs 可作为后续较重档位。32 CTA 两档时长明显更长，可用于独立压力对照。每个条件只有一份 P2 采集，五次来自同一进程。结果仅支持选择候选探针，不足以推断任何真实 QKV 的并发干扰或同 SM 共驻。
