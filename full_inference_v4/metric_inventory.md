# 指标可用性与证据范围

只执行了版本、帮助和设备/指标集 **help 查询**。没有运行 profiler target，没有
NCU 注入，没有模型加载。原始输出见 [cpu_environment_audit.json](analysis/cpu_environment_audit.json)。

当前 NSYS 2026.1.3 支持 `gb20x`、`gb20x-top` 和 Graph/Node 选择。NCU 不在默认
PATH，已在 `/opt/nvidia/nsight-compute/2026.2.1/ncu` 找到；version/help 成功。
此结果仅证明二进制可访问，不能推翻历史注入失败或证明权限/计数器可用。

[metric_availability.csv](analysis/metric_availability.csv) 从历史实际 CSV 提取了 25 个
指标名称，并列出其他机制指标的缺失状态。频率等历史异常字段即使存在也不作为
有效时钟证据。V4.1 当前没有任何新硬件样本。

| 指标/来源 | 当前证据 | 下一步 |
|---|---|---|
| SMs Active、SM Issue、Tensor Active、Compute Warps、DRAM R/W | 历史 NSYS 设备级采样；本机帮助支持原集合 | Gate 2 重新取得样本并确认定义、单位、覆盖与扰动 |
| Kernel 时间、Grid/Block、Context/stream/Graph/correlation | 历史导出 Schema；新分析器支持已知形式 | 新 SQLite 实际字段与身份验收 |
| Registers、Static/Dynamic SMEM | Launch 元数据依版本可得 | 检查实际字段，不把分配量称为利用率 |
| FP32/INT Pipe、动态 MMA 数量、L2 命中、Warp Stall、Eligible Warp、实际 Occupancy | 尚无可靠当前证据 | NCU 可用时仅重点 Kernel；失败则保留 UNAVAILABLE |
| 理论 CTA/Occupancy 上限 | 可依据实际 Launch 和 SM 资源进行必要条件分析 | 考虑分配粒度/其他限制；不冒充实际共驻 |
| GPU 时钟/功耗/温度 | 未来监督器 nvidia-smi 原文采样 | 稀疏样本不能用作高分辨率能耗积分 |

官方定义参考：
[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)、
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。
网站新版本出现的 NVTX Graph 投影参数不直接假定本机支持，必须以安装帮助与实测为准。

P3 必须保存 Replay、Cache Control、缓存清理及序列化条件。默认缓存清理可能改变
原应用 cache priming；动态计数与静态 SASS 指令列表也不能混同。失败最多进行
明确限定的诊断，禁止无限重试。Gate 0/1 不执行这类 GPU 诊断。

缺失使用 UNAVAILABLE/UNVERIFIED/NOT_APPLICABLE。0 仅用于确实测得的零或真实计数，
不能填充不存在的硬件指标。缺少数据时不输出瓶颈结论。
