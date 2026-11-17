# Occupancy API准备与不可替代的实测

当前没有调用任何实际Occupancy或函数属性API。`resource_model.occupancy_api_query`仅是延迟接口，在未来经过授权、Context和函数句柄合法时调用：

- `cuFuncGetAttribute`：NUM_REGS、SHARED_SIZE_BYTES、MAX_THREADS_PER_BLOCK、MAX_DYNAMIC_SHARED_SIZE_BYTES。
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`：实际函数、threads、dynamic SMEM对应的同类CTA容量。

API返回需同时保存Driver/Runtime版本、函数身份/代码Hash、UUID、Context、全部参数、返回值、cache/shared carveout配置及异常；当前表内occupancy_api=`NOT_RUN_CPU_ONLY`。接口调用本身不收集这些所有环境信息，未来监督worker必须补齐记录，不能空填为VERIFIED。

同类函数Occupancy API不是两个不同Kernel共驻预测器，不能将两个API返回直接相加。理论Occupancy、实际Active Warps和Achieved Occupancy也不同。当前没有NCU新数据，不通过工具不可用而强行填数。

CUDA头文件完整副本只存本地ignored cache；公开粒度审计记录原始路径、版本条件、函数名、行号和SHA，便于从同版本toolkit复核。reservedSharedMemPerBlock与运行时资源属性缺失必须保留，不把cuobjdump静态1024B擅自解释为该值。
