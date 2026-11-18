# QKV 实机属性与混合 CTA 驻留模型

使用已通过输出验收的两种 QKV 单节点重放，以及四种独立 Probe 的函数属性。RTX 5090 上读取到 65536 registers/SM、102400 B Shared Memory/SM，以及每 CTA 1024 B 的 Driver 保留 Shared Memory。实测单函数 Occupancy API 为 NVFP4 QKV 1 CTA/SM、BF16 QKV 5 CTA/SM。Light 探针均返回 24 CTA/SM，Heavy 均返回 3 CTA/SM。所有这些返回值针对单一函数。[查询记录](../gate4_spatial/qkv_replay_validation/report.md) · [探针验收](../gate4_spatial/probe_gpu_validation/report.md)

将设备保留量及寄存器、Shared Memory 分配粒度应用到已取得的函数和启动属性。枚举 1–5 个 BF16 主 CTA 或 1 个 NVFP4 主 CTA 与单个 Probe CTA 的资源必要条件，保留 32/128 Grid 两个已采用配置。48 行是理论候选布局。配置的 Grid 影响其他 SM 的供给机会，单个 CTA 的静态资源不随 Grid 变化。[运行时属性模型表](analysis/live_qkv_resource_model.csv)

| 主任务 + 单 Probe CTA | 寄存器需求 | Shared Memory 需求 | 理论结果 |
| --- | ---: | ---: | --- |
| 1 NVFP4 QKV + FP32 Light | 65280 | 90112 B | 总量通过，四寄存器分区模型无放置 |
| 1 NVFP4 QKV + FP32 Heavy | 68608 | 122880 B | 两项都超过 SM 总上限 |
| 1 NVFP4 QKV + BF16 Tensor Light | 66048 | 91136 B | 寄存器超过 SM 总上限 |
| 1 BF16 QKV + FP32 Heavy | 8192 | 51712 B | 通过已知必要条件 |
| 4 BF16 QKV + FP32 Heavy | 20480 | 105472 B | Shared Memory 超过 SM 总上限 |

四分区结论来自当前 CUDA 头文件模型。Source/Driver 元数据没有给出实际每个 CTA 被分到哪个 SM。实测 GPU Kernel 重叠仅在 FP32 Light 两档已核验。该证据不能用来填充 Heavy 和 Tensor 的并发记录。所有同 SM 共驻状态仍为 `UNVERIFIED_NO_DIRECT_CTA_POSITIONS`。

下一组拟用相同 FMA 算术任务比较 Light/Heavy，同一 Grid 下固定输出数与操作数。Heavy 每 CTA 使用 128 线程、更多寄存器和实际访问的 32 KiB staging，共享内存和计算指令调度会一起变动，不能将性能变化归因给单一资源。若 Heavy 与 NVFP4 主 Kernel 有 GPU 时间重叠，模型和区间可支持跨 SM 或 CTA 时间交错的解释。若没有重叠，需要先排查实际 GPU 起跑、Stream 依赖和工作持续时间。Gate/Up 的更大 Grid 提供同一投影类型下的内部对照，需独立核验原始回放模型和输出。
