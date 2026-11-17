# 资源驻留阶段报告：CPU模型及边界


本次交付为 **CPU 侧审计、实现与审查材料**。没有加载模型权重，没有初始化 CUDA Context，没有运行 GPU Kernel、模型推理、NSYS 或 NCU。CUDA 离线编译与反汇编只使用 CPU，不等于 GPU 正确性验收。

历史 Gate 2/3 保持原始测量范围；本次没有新的并发性能结果。`analysis/` 中没有数据行的表是预留结构，不能视为零延迟、零重叠或失败实验。`VERIFIED` 必须同时注明验证对象和范围，CPU 测试通过不能提升 GPU 证据等级。


已从历史数据提取8个主CTA配置，取得4种Probe离线编译资源，生成8行原始同类CTA上界和56行混合资源模型；4项CPU测试通过。未调用活Occupancy API，没有任何实际同SM证据。

最值得带入后续试验的预测是：NVFP4主CTA占用64512个静态寄存器下界；FP32 Light即便通过全局加法，也不通过当前4子分区模型；Tensor Light和Heavy已受到已知总资源下界限制。BF16单CTA与Probe更容易满足必要资源条件，但实际BF16可能有多个CTA驻留。

这个预测使“NVFP4小Grid有跨SM机会”与“NVFP4大CTA难以同SM共驻”成为可以同时成立的假设，并未证明任一实际结果。

| 问题 | 当前回答 | 状态 |
| --- | --- | --- |
| NVFP4大CTA限制同SM余量吗 | 已知静态资源模型支持限制，运行时待核验 | PARTIALLY_SUPPORTED（CPU模型） |
| BF16更容易与Heavy共驻吗 | 单CTA必要条件更宽松，实际驻留数量未知 | 实际结论UNVERIFIED |
| Light能否进入NVFP4所在SM | FP32 Light分区模型不通过，无直接观测 | SAME_SM_CO_RESIDENCY_UNVERIFIED |
| 主要资源限制 | 当前模型为寄存器/分区，Heavy亦受SMEM限制 | 模型内VERIFIED，运行时UNVERIFIED |
| 模型与实际是否一致 | 尚无实际混合CTA数据 | UNVERIFIED |
| 不能同SM时能否空间并发 | 需要Spatial时间线与性能 | UNVERIFIED |
| 资源与损失关系 | 无新P0并发观测 | UNVERIFIED |

剩余关键证据：runtime reserved shared、实际function属性、当前设备/API返回、缓存/carveout、真实Kernel起跑/重叠、主CTA位置。后续即使API与模型吻合，仍不能直接称为同SM实测。
