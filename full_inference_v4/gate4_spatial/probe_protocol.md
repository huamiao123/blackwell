# 辅助 Probe 定义与验收

Probe A 使用明确的 FP32 `fma.rn.f32`，FP32 输入和寄存器中间状态、输出数组依赖全部计算。Probe B 使用 BF16 WMMA，FP32 累加，实际 A/B 操作数写入并读取分块 Shared Memory；不是普通标量 GEMM。

离线 nvcc 13.3 编译到 SM120 CUBIN，cuobjdump 的静态结果：

| symbol | registers_per_thread | static_shared_bytes | ffma_static_instructions | bf16_mma_static_instructions | gpu_correctness |
| --- | --- | --- | --- | --- | --- |
| fp32_light | 22 | 1024 | 4 | 0 | UNVERIFIED |
| fp32_heavy | 32 | 1024 | 16 | 0 | UNVERIFIED |
| tensor_light | 43 | 1024 | 0 | 2 | UNVERIFIED |
| tensor_heavy | 40 | 1024 | 0 | 2 | UNVERIFIED |

Tensor 指令为 `HMMA.16816.F32.BF16`。静态 FFMA/HMMA 条数不是动态执行量，也不代表 GPU 输出正确。编译器报告的 static shared=1024B 原样保存，不能擅自去掉或当成已查明的 runtime reserved shared。

| 配置 | Threads/CTA | 计划 Dynamic SMEM | 固定工作量 |
| --- | ---: | ---: | --- |
| FP32 Light | 32 | 0 | 262144 items ×16 values ×512 FMA，4,294,967,296 FLOP |
| FP32 Heavy | 128 | 32768 B | 同一逻辑 FMA 数，实际使用 staging |
| Tensor Light | 32 | 1024 B | 4096 个16×16 tile，每 tile 256次16×16×16 MMA，8,589,934,592 FLOP |
| Tensor Heavy | 128 | 32768 B | 同一逻辑 MMA 数，多 warp、不同 staging |

Grid-stride 分配固定逻辑单位，Grid 不改变总计算量。增加迭代数会延长工作，不等于增加瞬时资源压力。Light/Heavy 资源以编译结果为准，而不是设计目标；FP32 Heavy 实际只有32寄存器/线程，其较大SMEM仍形成不同压力。

CPU oracle 使用有界、可精确追踪的二进制分数和完整结果验证定义，检查重排/不同 staging 的预期不变量。下一次授权 GPU 试点必须先在各 Grid 完整校验输出，记录实际启动动态 SMEM、cuFuncGetAttribute、输入/输出 Hash、实际工作量和独立延迟；任一失败不得进并发。

离线编译原始日志、SASS、资源、命令和 SHA 都在 `analysis/offline_compile/`。没有使用空循环冒充计算，没有生成 DRAM Probe；B 的读写仍存在，不能称为完全零访存。
