# CTA资源模型与来源

主CTA资源从Gate 3的operator_timing/precision_comparison/validation读取；不是重新查询GPU。8行覆盖两精度的QKV/O/GateUp/Down。四类Probe资源来自当前CUDA源码的离线CUBIN，动态SMEM是计划启动值，actual_dynamic_shared明确未运行。

历史硬件：170SM、65536 registers/SM、102400B shared/SM、32 threads/warp、48 warps/SM、24CTA/SM。1536 threads/SM是48×32推导，未冒充活设备属性。

最简单下界为 R=threads×regs/thread、S=static+dynamic、W=ceil(threads/32)，再检查CTA/threads/warps/单Block限制。NVFP4原始R=64512，BF16=3968；忽略粒度的同类CTA上限分别1和6。这个数不是Occupancy API返回，也不是实际驻留。

`cuda_occupancy.h`（CUDA13.3）case12规则经CPU读取并保存SHA：每warp寄存器分配粒度256、shared粒度128B、4个子分区。模型先对每warp寄存器向上取整，再乘warp数；SMEM进行相应取整。原头文件还考虑runtime reservedSharedMemPerBlock，此属性在历史库存缺失，本模型不假设它为0或已知，而将正向可行结果降为必要条件通过。

混合资源模型检查 nA×R_A+nB×R_B 等总量及每Block限制。正向标签 `NECESSARY_BOUNDS_PASS_NOT_SUFFICIENT`；负向是已知资源必要下界超限。仍需考虑寄存器分区、运行时函数资源、carveout及动态行为。

## 为什么简单加法会误导

NVFP4主CTA有12warps，每warp168×32=5376 registers，已经是256整数倍。若按4个16384-register分区放置，每分区最多3个主warp；放完12warp后各分区仅剩256。

FP32 Light实际22regs/thread，一个warp按粒度需要768 registers。全SM加法64512+768=65280不超过65536，但在上述分区模型下没有位置放一个768-register warp。`partition_model.py`穷举允许的分配，并保留 `REGISTER_PARTITION_MODEL_NO_PLACEMENT`。

这是CUDA头文件参数下的CPU模型预测，并非实际混合Occupancy API或同SM排除实测；静态寄存器报告、架构动态行为和调度仍需GPU核验。不能将模型输出升级为DIRECT_SAME_SM_MAPPING_VERIFIED。

Tensor Light实际43regs/thread，按粒度1536/warp，64512+1536=66048，连简单全局下界也超限。Heavy类计划32KiB动态SMEM，与NVFP4的88064B相加超过SM容量，同时寄存器也超限。BF16单CTA与四类Probe均通过当前必要条件；当多个BF16 CTA已经驻留时结论会改变，因此保存nA=1..6的组合。

QKV和Gate/Up主CTA资源相同但Grid不同；理论每SM模型相同，实际跨SM并发仍可能不同。Grid不是这个资源模型的Occupancy分母，也不提供实际CTA位置。
