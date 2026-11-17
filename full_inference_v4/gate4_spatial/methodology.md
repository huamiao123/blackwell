# 空间并发实验方法


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


## 假设与可证伪条件

H1：NVFP4 小 Grid 的 QKV 可能较 BF16 更容易与辅助任务并发。H2：同一辅助工作量产生的主任务干扰和辅助吞吐保持率随精度变化。H3：QKV 与 Gate/Up 的行为可能不同。任何一项均可被实验否定，CTA 数量不是可利用资源的证明。

Gate 3 数据用于选择实际主函数与逻辑算子，不用于充当本轮 A Alone 基线。新 A 必须来自同一活模型和活 Graph 的实际节点，所有指针、Context、模块生命周期均在本轮验证。历史参数 SHA 仅是身份审计记录，不是新进程中可执行的地址。

## 计时与工作量

P0 用已准备完成输入上的 CUDA Event 包络测量诊断 Graph，A、B、实测 Serial AB/BA、Parallel AB/BA 的测量范围一致。该包络含 Graph 派发等成本，不能称为精确单 Kernel 时长。P2 单独采集实际 CUPTI Kernel 区间、Stream、Context、Correlation 和 UUID；不能把 P2 耗时替代 P0。

串行条件是真实排队执行两个任务，用已记录的结束 Event 建立依赖。禁止将两次 Alone 的时间相加当串行实测。Work Hash 包括 A/B 定义、调用次数和缓存协议，并分别核查实际完成的 A/B 单位数量与输出。A Alone 完成 A、B Alone 完成 B；Serial/Parallel 必须完成同一 A+B。

主任务损失 `I_A=T_A_parallel/T_A_alone-1`。相同 B 完成量下吞吐保持率 `R_B=T_B_alone/T_B_parallel`。联合收益 `S=T_serial_measured/T_joint_parallel`。联合边界从统一释放事件到两分支均完成。B 尾部不在 A 内完成的部分不能计作已利用主任务资源；缺少块级进度时不按时间比例推算完成工作。

时间重叠取 `max(0,min(A_end,B_end)-max(A_start,B_start))`，重叠比例除以两 Kernel 时长的较小值。多 Kernel 回退须使用区间并集，不能套用单 Kernel 对公式。没有 CTA 位置证据时一律保留 `SAME_SM_CO_RESIDENCY_UNVERIFIED`。

## 公平缓存与准备

QKV 单层权重可能命中 L2。连续重放一个节点不能描述为真实整模型的权重流式状态。优先设计与实际 Decode 匹配的准备前缀，在所有条件保持一致，将其放在测量前并记录完成 Event；同时核查其本身是否造成热缓存。若无法建立匹配状态，实验标记为 `WARM_DIAGNOSTIC_REPLAY`，只研究该条件的并发，不外推完整推理收益。

不以反复读同一权重模拟整模型，不随精度改变缓存清理策略；不以人为全设备同步跨越真实 Prefill/Decode 交界。诊断 Graph 的准备同步不属于完整推理实验路径。

## 起跑偏差控制与重复

同一 Context 的两条非阻塞 Stream 先等待设备内存标志。A/B 均提交后，控制 Stream 记录 joint_start 并释放标志，最后等待两者的结束 Event。这减少 CPU 提交先后偏差，但不保证 GPU 同时起跑；实际起跑差必须由 P2 记录。未记录 Event、SM 自旋或 CPU Sleep 均不作为起跑门。

至少五个有效配对 Block。交替 BF16/NVFP4，AB/BA 顺序平衡，九种条件在冻结种子下随机化。原始数据和失败均保存，不能把调用数当独立样本。关键结果另做独立进程复验。初始统计器支持 5–10 个 Block 的 t 区间，须注明条件独立性假设；不能伪称跨进程或跨设备总体区间。

## 停止与证据

当前没有 GPU 授权。未来必须复用历史监督逻辑：先校验窗口，再检查 GPU 和其他用户控制器，仅运行一个有限任务，硬超时，冲突只清理自己的进程。没有空闲轮询，不改驱动/功率/时钟，不自动推进下一 Gate。

活节点输入不明、输出未校验、负对照重叠、错误设备/Context、工作量不等、空 Trace 或其他用户进入时停止。直接同 SM 共驻不在当前证据能力内。
