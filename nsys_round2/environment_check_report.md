# NSYS第二轮环境与既有资产检查

本机NSYS版本2026.1.3.425。授权包装器可列出设备0：RTX5090 GB202、PCI0000:27:00.0；可用集合为gb20x及gb20x-top，正式选择gb20x。参数帮助确认graph/node granularity、cudaProfilerApi捕获、stop-shutdown、10kHz采样受支持。

历史prof/gpumetrics_{bf16,nvfp4}.sqlite内确有GPU_METRICS、TARGET_INFO_GPU_METRICS、CUPTI_KERNEL、RUNTIME、NVTX、GPU及CUDA_DEVICE映射；字段已用PRAGMA核对。历史实测样本间隔中位数约1ms，历史脚本显式设置1000Hz。旧DECODE_ONLY标记只覆盖33-token生成且包含首步，因此重新限定本轮Decode窗口。

复用原decode_prof2.py/decode_graph.py的LLM启动路径，新增的是请求计数、实际调度观察和标记，不是模型或Kernel优化。正式EngineCore为同进程InprocClient；Graph有实际Replay及cudaGraphLaunch记录，Eager没有。NVFP4实际Kernel名确认进入SM120 E2M1 block-scaled CUTLASS路径。两模型Attention均固定FlashInfer XQA、FP8 KV。

所有命令和实际输出存environment.txt及logs/environment_*.txt；historical_schema.json保存旧库Schema。正式逐库Schema和身份/时间范围保存在analysis/*_validation.json。授权包装器接受所需NSYS参数；TMPDIR单独preserve可用，而同时preserve TMP/TEMP被sudo拒绝，该探测没有启动模型。应用缓存和脚本输出定位在项目内；NSYS日志仍显示工具内部/tmp会话暂存，未声称该内部路径已成功重定向，也未在项目外手动清理。

未调用NCU。未变更驱动、权限、GPU频率或功率限制。当前条件下Eager tracing有明显扰动，其后续解释见final_report.md，不视作正常性能画像。
