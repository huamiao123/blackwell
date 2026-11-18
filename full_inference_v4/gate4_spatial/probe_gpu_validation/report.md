# RTX 5090 辅助探针 GPU 验收

2026-11-18 20:00:11–20:00:14（北京时间），在北交共享 RTX 5090 上完成一次独立探针验收。FP32 FMA 和 BF16 Tensor Core MMA 各有 Light、Heavy 两种版本，每种在 32 和 128 CTA 下运行。八组都通过全量输出校验。输入是确定性合成数据，全程未加载模型权重。

使用 Gate 4 先前离线编译的 SM120 CUBIN，SHA256 与编译清单一致。每组运行一次预热和五次记录。每次启动前把输出填成 NaN，结束后把全部结果复制到 CPU 与参考值比较。同一配置的两种 Grid 使用相同的逻辑工作量和输入，五次记录的输出哈希一致，两 Grid 的输出哈希也一致。CUDA Event 覆盖单次探针启动，输出填充和回读在计时区间外。五次记录来自同一进程，主要用于独立探针校准。

| 探针 | 每 CTA 线程 | 寄存器/线程 | 动态 Shared Memory | 单函数 Occupancy API blocks/SM | 32 CTA 中位数 | 128 CTA 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 Light | 32 | 22 | 0 B | 24 | 5150.272 µs | 1311.264 µs |
| FP32 Heavy | 128 | 32 | 32768 B | 3 | 528.960 µs | 157.664 µs |
| BF16 Tensor Light | 32 | 43 | 1024 B | 24 | 981.376 µs | 258.912 µs |
| BF16 Tensor Heavy | 128 | 40 | 32768 B | 3 | 1073.888 µs | 279.232 µs |

四种函数的 `cuFuncGetAttribute(SHARED_SIZE_BYTES)` 均为 0 B，而离线 `cuobjdump` 资源表的 `SHARED` 均为 1024 B。这两个来源的字段在本次实际不一致。运行时模型取函数属性 0 B，再加 `cuDeviceGetAttribute(RESERVED_SHARED_MEMORY_PER_BLOCK)` 实测的 1024 B。样本和元数据见[原始结果](raw/probe_20261118_120007_utc/result.json)，运行区间和命令见[监督日志](raw/probe_20261118_120007_utc/supervision.json)，对应版本见[源码快照](source/acceptance)。Occupancy API 返回值描述单一函数在给定启动参数下的理论容量。[NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__OCCUPANCY.html)

把实测保留量代入 CUDA 13.3 头文件的分配模型，结合 Gate 3 主 Kernel 启动元数据，可得 BF16 投影每 CTA 的 Shared Memory 模型占用为 17920 B，NVFP4 为 89088 B。按 102400 B/SM 的容量分配，**BF16 仅按 Shared Memory 的理论上限从 6 降为 5，NVFP4 仍为 1**。这一步用的是历史主 Kernel 元数据和当前设备查询。主 Kernel 的活函数 Occupancy API 尚未查询。见[逐投影计算表](analysis/primary_smem_bounds.csv)。

Gate 3 的 QKV 并集时间除以 9216 次调用约为 NVFP4 19.5 µs、BF16 32.6 µs。这个数字用于粗定探针时长，来自历史 P2。当前 FP32 Light 在 32/128 CTA 下要 5.15/1.31 ms，容易使并发试验的联合时间由辅助尾部主导。下一版已把校准点设为 65536 items、8 或 32 轮，并在每个轮数下固定工作量对照 32 和 128 CTA。各 Grid 保持所有 CTA 有工作，实际时长待测。见[校准配置实现](../src/validate_probe_gpu.py)。

20:02:17，第二轮校准前的 `gpu-who` 显示 0% 利用率，监督器随后检测到其他用户的自动实验控制器并在启动 worker 前拒绝执行。未等待控制器结束，也未启动第二轮 GPU 负载。首轮结果保留，第二轮只有[阻止记录](raw/probe_20261118_120214_utc/supervision.json)。

22:01 重新获准使用空闲设备后，短探针校准的四组均通过全量输出校验，CUDA Event 中位数为 52.8–156.1 µs。后续以实际 GPU Kernel 区间核验是否与 QKV 的时间尺度匹配。[短探针校准报告](qkv_calibration.md)

- **VERIFIED**。八组独立探针输出，探针实际函数属性，单函数 Occupancy API 及设备保留 Shared Memory。
- **模型推导**。使用运行时设备保留量修正 Gate 3 主投影的 Shared Memory 容量上限。
- **UNVERIFIED**。短探针时长，真实 QKV 节点安全重放，A/B GPU 重叠和干扰，同 SM 共驻。后续需要在可用窗口校准短探针并核验真实 QKV 的输入和输出，再比较实测串行与并发。
