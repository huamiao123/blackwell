# 真实 QKV 重放与首轮并发时间线

在北交 RTX 5090 上复现了 Gate 3 的第 18 层 QKV 主 Kernel，并对 BF16 和原生 NVFP4 各采集一个并发配对块。每种精度只读取对应 checkpoint 的该层 Q/K/V 权重及必要 scale。两边使用同一份 8×4096 BF16 合成激活，输出均为 8×6144。NVFP4 使用原 vLLM 的 ModelOpt scale 规则和 CUTLASS 权重处理，激活量化在诊断 Graph 之外完成。BF16 复用 vLLM 的原始 unquantized dispatch。[历史指纹](gate3_reference.json) · [验收源码](../src/validate_qkv_replay.py)

私有 CUDA Graph 各只有一个 Kernel。Driver 查询得到的函数符号、Grid、Block、动态 Shared Memory 和寄存器数与 Gate 3 匹配。参数中绑定的是本进程第 18 层的实时权重引用。每次先用 NaN 覆盖输出，再重放 Graph，三次全量输出均与该精度原 vLLM 投影路径逐元素一致。该校验确认单节点分离的数值等价性，量化模型与 BF16 的输出未作跨精度精度比较。[NVFP4 单独验收](raw/probe_20261118_141906_utc/result.json) · [BF16 单独验收](raw/probe_20261118_142102_utc/result.json)

| 主 Kernel | CTA/次 | Threads/CTA | Registers/thread | Dynamic SMEM | 单函数 Occupancy API |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 SM120 CUTLASS | 48 | 384 | 168 | 88064 B | 1 CTA/SM |
| BF16 CUTLASS WMMA | 384 | 32 | 124 | 16896 B | 5 CTA/SM |

配对试点于北京时间 22:24–22:25 完成。两边使用相同 FP32 Light 探针，128 CTA、每 CTA 32 线程、65536 个逻辑项、8 轮、16777216 次浮点操作。两个私有 Graph 在同 Context 中使用独立 CUDA Stream。设备内存标志在两侧提交后放行。九个条件均实际执行并检查输出，Serial AB/BA 通过已经记录的 CUDA Event 建立结束依赖。[配对试点源码](../src/qkv_overlap_pilot.py) · [NVFP4 原始记录](raw/probe_20261118_142409_utc/result.json) · [BF16 原始记录](raw/probe_20261118_142458_utc/result.json)

NSYS 的 NVTX 标签经 launch correlation ID 与 GPU Kernel 区间关联。每种精度均核验 15 个带试点标签的 Kernel，主投影和探针各自保持同一 Graph 节点身份，GPU UUID、进程、Context、函数符号及启动属性已核对。每种精度的 Serial AB、Serial BA、同提交 Stream 对照、不同 Stream 的显式无重叠对照均为 0 µs 重叠。NSYS Kernel 活动的 Stream ID 与提交时的 Stream 句柄使用不同标识，提交关系由队列协议给出，实际重叠由 GPU 时间戳验证。[GPU 区间](analysis/qkv_pilot_intervals.csv) · [计算汇总](analysis/qkv_pilot_summary.csv)

| 主任务 | 顺序 | GPU 重叠 | 占较短 Kernel | A Kernel 时长变化 | B 吞吐保持 | 实测串行/并发联合 Event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NVFP4 QKV | AB | 20.831 µs | 96.87% | +14.67% | 76.05% | 1.621× |
| NVFP4 QKV | BA | 20.896 µs | 96.88% | +15.02% | 76.05% | 1.479× |
| BF16 QKV | AB | 19.616 µs | 96.38% | −5.63% | 82.86% | 1.733× |
| BF16 QKV | BA | 20.928 µs | 97.76% | +0.59% | 78.77% | 1.624× |

A 与 B 分别相对同一份 P2 的 A Alone、B Alone 计算。A Alone 的 GPU 时长为 NVFP4 18.752 µs、BF16 21.600 µs。B Alone 约为 16.864 µs。表中联合时长比使用同顺序的实测串行和并发 Event 包络，每组完成同一 A+B 工作量。该指标包含放行和汇合，和两个 Kernel 各自的执行时间边界不同。

这一轮说明两个真实 QKV Kernel 均可与探针并发。在这个固定配置下，NVFP4 并未表现出更低的干扰，主任务多用时约 15%，已超过后续实验拟用的 5% 预算。BF16 AB 的负值只有一次测量，不能解读为并发让 BF16 加速。每种精度仅一块 P2，反复使用相同投影缓冲区，实际 L2 状态尚未测量，结果不能外推到完整 Decode。后续需至少五个无 Profiling 配对块、精度顺序交替及独立进程复验，再判断相对差异。CTA 级位置仍未观测，同 SM 共驻和完整推理收益均未验证。

前两次 NVFP4 验收分别因独立层对象缺 `input_size_per_partition`、Occupancy API 返回值包装调用错误而停止。两项均已修复，失败记录保留。[第一次失败](raw/probe_20261118_141752_utc/worker.log) · [第二次失败](raw/probe_20261118_141836_utc/worker.log)
