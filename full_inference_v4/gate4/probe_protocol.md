# 辅助探针

FP32 Light/Heavy 按相同总 items 和轮数分配工作，Heavy 使用实际消费的共享内存数据。BF16 Tensor Core 探针使用 BF16 矩阵片段和 FP32 累加，验证 SASS 指令与完整输出。Grid 扫描保持总逻辑工作固定。不同探针需要各自的 B Alone。

DRAM 单因子探针采用 [Read、Write、Copy 六个独立符号](../gate4_spatial/src/dram_streaming.cu)。Light 为 32 线程，Heavy 为 128 线程并对 32 KiB 暂存 tile 进行跨 Warp 读写。每次运行遍历互不别名的完整缓冲区，单侧数据区建议至少为设备 L2 的 4 倍。Read 校验每 tile 的模 (2^{32}) 加权和，Write 校验固定种子输出，Copy 校验全量一致。报告逻辑字节与硬件测得字节，需要先用实际设备计数验收访存路径。第一轮空间并发未使用 DRAM 探针。

已有探针结果和资源字段见 [GPU 验收](../gate4_spatial/probe_gpu_validation) 和 [原实验协议](../gate4_spatial/probe_protocol.md)。六个流式符号已通过 SM120 离线编译及内存指令检查，尚待 GPU 正确性和真实 DRAM 流量验收。[流式探针编译结果](../gate4_spatial/analysis/offline_compile/dram_20261119/dram_probe_validation.csv)
