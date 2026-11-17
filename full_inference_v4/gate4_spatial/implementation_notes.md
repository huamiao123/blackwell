# 实现、生命周期与未解决问题

## 独立诊断 Graph

`replay.py` 通过活节点 `cuGraphKernelNodeGetParams` → 新建 Graph → `cuGraphAddKernelNode` → `cuGraphKernelNodeCopyAttributes(dst,src)` 建立独立副本，不修改 Gate 3 原 Graph。源 Graph 活着时复制参数，模型和参数所有者保持强引用，最后一个 join 完成前不释放副本。

复制参数值不代表其指向的数据已准备好。`LiveKernel` 要求当前 PID/Context/UUID、输入就绪、生产依赖完整、输入不变、输出区域合法、参考输出通过、执行签名一致、缓存协议、可变状态重置和非 Kernel 依赖证明。任一项缺失均拒绝实例化/执行。

尚未实现可直接验收的 vLLM 激活输入快照/重置/输出 oracle 运行时适配器。Gate 3 的模块权重绑定只能识别节点，不能自动解码其全部 CUTLASS/编译器参数结构或推断激活缓冲区生命周期。`ancestor_closure` 只生成 CPU 前驱计划；非 kernel/event/memcpy 等前驱会标记需审查，不能假设删掉前驱仍等价。

建议下一步在明确授权的诊断进程内：捕获活模型与节点，核实函数参数 ABI 和读写范围，建立生产前缀或合法快照，参考原路径输出，验证副本的函数/形状/资源/输出一致，再开始性能试点。上述动作本轮没有执行，不能描述为“单节点重放已经可用”。

## Stream 与阶段回退

`stream_protocol.py` 实现九条件队列逻辑；`cuda_stream_backend.py` 是延迟构造的 Driver 适配组件，需要外部有效 permit、活 Context 和已记录输入 Event，尚未进行 GPU 集成验收。必须在未来 supervisor 内调用，不能直接导入组件视作具备授权。

若不能安全重放目标节点，`fallback_stage.py` 可包装已有 `execute_model`，按真实调度行选择 Prefill/Decode。它只提供阶段级回退接口，不能声称与某一 QKV 独占重叠。空调度和 Mixed 不冒充 Decode。当前无实际集成时间线。

## 分析与测试

`analyze_trace.py` 先核对 SQLite Schema、实际 UUID 与 CUDA device 映射，利用 CPU NVTX 内的 CUDA Runtime correlation 关联异步 GPU Kernel，不用 CPU NVTX 边界直接裁剪 GPU。只对一 A/一 B 求单 Kernel 重叠，跨 Context、多目标、同 Stream 假并发及负对照重叠拒绝。实际主模块身份仍需独立活节点证明。

CPU 测试发现并修复了异常路径 SQLite 未关闭的问题。早期离线反汇编后处理曾因 `Function :` 与 `Function name:` 格式不同失败；已修复两种解析并重新离线编译。这些是准备过程故障，不是 GPU 实验失败。

## API 依据

CUDA Graph 的参数复制、属性复制和对象生命周期依据 [Driver Graph API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)。函数属性依据 [Driver Execution API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html)。文档与 CPU fake backend 不能代替当前 CUDA Python 绑定的真实 GPU 集成验收。
