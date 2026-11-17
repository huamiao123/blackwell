# 当前阶段结果

V4.1 Gate 2 的 E1/E2 基础采集已完成。详细结果见
[gate2_report.md](gate2_report.md)，包含正式性能、四张图、Profiler扰动、
执行路径、设备级指标、UNKNOWN归属缺口与失败记录。

完成40次P0完整推理、四份P1 GPU Metrics和四份P2 Node Trace。
Batch=1完整模型均值：BF16 2.727秒，NVFP4 1.762秒；
Batch=8：BF16 4.059秒，NVFP4 2.143秒。
这是预编码离线模型窗口，不包含在线网络和本次Tokenization。

指定负载性能差异为VERIFIED；硬件资源变化为PARTIALLY_SUPPORTED，
每条件仅一份Metrics采样。资源可利用性RQ2、真实收益RQ3仍为UNVERIFIED。
没有进行辅助Kernel并发、NCU或Gate 3算子归因。

GPU工作结束时已关闭窗口，无实验进程和permit残留；不自动进入下一Gate。
CPU准备历史见[implementation_report.md](implementation_report.md)与
[analysis/quality_validation.json](analysis/quality_validation.json)，其“未运行GPU”
仅指原准备快照，不代表当前Gate 2状态。
