# RTX 5090（SM120）BF16 vs 原生 NVFP4：资源瓶颈迁移与协同收益实验——V2 技术实施版

**日期：2026-11-14｜版本：V2.0｜用途：交给具备 CUDA/CUTLASS/NCU 能力的工程师或 Codex 实施。**

> **研究问题**：在同一张 RTX 5090 上，BF16 → 原生 NVFP4 是否改变矩阵乘法及真实 LLM 推理的资源瓶颈与干扰特征？能否利用这种差异让两个有用的任务共同完成得更快？
>
> **原则**：第一阶段寻找差异、第二阶段进行因果归因、第三阶段验证实际资源回收。允许“没有额外收益”的负结果；不允许根据低 Occupancy、低 Tensor Pipe Active 或理论 FLOPS 直接宣称存在可被使用的空闲资源。

## 0. 对 V1 的逐项修订（必须先读）

| V1 或此前讨论中的问题 | V2 的修改及理由 |
|---|---|
| 把 BF16 / NVFP4 两个库 Kernel 直接相减，称为“精度类型本身导致差异” | 分为**硬件指令能力、相同量化数值的路径对照、真实端到端**三层。不同 Kernel 实现构成混杂因素，必须显式记录并通过受控微基准辅助归因。 |
| 将“NVFP4 权重+BF16 计算 → NVFP4 原生计算”的差值视为纯原生指令收益 | 改为 G0–G6 对照。在线反量化、激活转换、融合方式和布局也会改变；不能认为任何两组能完全剥离单一因素。 |
| 只说“原生 FP4”，但未严格检验量化、Scale、指令 | 增加 E0 的 **SM120 `mma.sync.aligned.block_scale` 证据、同源输入、正确的 NVFP4 block/global scale、布局及负例检验**。 |
| 用 CUTLASS 79a 成功运行就认为它是同源 BF16→FP4 数据的公平基线 | 79a 官方示例默认**独立随机生成 E2M1 payload 和 Scale**，是指令通路 smoke test；必须改造数据准备及数值参考才是量化对照。 |
| 对 NCU Cache、Replay 和并发的使用说明不够细 | 区分 CUDA Event 正常运行计时、NCU **单 Kernel 分析**、Nsys 并发时间线；必要时用 **Range Replay / Application Range Replay** 做并发窗口的聚合计数器，不能默认用普通 Kernel Replay 证明并发。 |
| 用一次“extra MMA”耗时不变说明算力空闲 | 改为**独立受控 MMA 微基准 + SASS 指令审计 + 同 SM 驻留检查 + 真实双任务进度**；只复用寄存器里的旧操作数不代表可运行独立任务。 |
| L2 热/冷仅靠 `--cache-control none` | 缓存状态由应用主动制造，读回 NCU 的 DRAM/L2 计数验证；NCU 参数仅用来避免工具意外清缓存。**数据在 L2 里**不是根据名义容量就能证明。 |
| 并发实验只看 `T_serial/T_joint` | 增加两个任务各自完成的实际工作量、双侧减速、绝对吞吐、Pareto 边界、NSYS 验证重叠、功率/时钟对照。 |
| 将主 GEMM “每个 kernel 测 100 次”与 100 个独立样本混为一谈 | 计量层次分成一次 graph 内 K 次调用、一次完整测量样本、独立进程/会话；bootstrap 对独立样本或会话做聚类，不能把 K 个节点当成 K 个独立样本。 |
| 把前期研究限定到 KV FP4、低秩修正或 Prefetch | 不预设机制。**先定位 BF16/NVFP4 差异**，最终用真实任务检验。 |
| 理论峰值直接套入利用率 | 统一按照**稠密、指令精度、累加类型、有无稀疏**比较；实际占用以本机测得的峰值/计数器为主。 |

---

## 1. 整体实验结构和进入下一阶段的条件

```
E0 设备/工具链/原生指令/量化正确性 ───────► GATE-0
 │
 ├─ E1 单 Kernel 实测资源画像（BF16/NVFP4）──► GATE-1
 │
 ├─ E2 计算与数据供给的受控敏感性诊断 ─────► 选定 1–3 个可证伪瓶颈假说
 │
 ├─ E3 双任务干扰 + 真正并发 + Pareto ──────► GATE-2
 │
 └─ E4 真实 FFN/Transformer 子图的有用任务回收 ► GATE-3
```

- **GATE-0**：FP4 确认为 SM120 本机原生 MMA；BF16/FP4 数值自检成立，记录 Scale 语义/布局、输出不全 0、不含异常 NaN。
- **GATE-1**：E1 每个主 M 有合规时间统计和多个硬件指标；至少知道哪一段是 GEMM core、哪一段是动态激活量化等前处理；不得仅报告“NVFP4 比 BF16 快 x 倍”。
- **GATE-2**：至少在两个工作点观察到可重复的差异响应，且通过缓存状态、Tile、寄存器、频率、竞争者工作量等控制排除简单解释；无显著差异则报告 no-go。
- **GATE-3**：至少一个*真实有用且依赖合法*的联合任务；优化后应击败相同精度下最强串行/融合/已有并发基线，而不是只击败人为拆开的弱基线。

### 必须交付物

源码、构建说明、锁定版本、正确性单测、每轮原始 JSON/CSV、`ncu-rep`、`nsys-rep`、SASS/PTX 检验、含环境与错误日志的报告、绘图脚本、可复现实验配置。**不得只有图片或口头结论**。

---

## 2. 机器与环境预检：先解决“测量是否可信”

### 2.1 服务器约束

用户服务器已确认有 RTX 5090，之前驱动曾显示 `595.99.02`；**服务器实时状态必须重新查询**。`nvidia-smi` 的 CUDA Version 是驱动可支持的 CUDA 接口版本，不代表 `nvcc`、CUTLASS、NCU 已经可用。不要在这台多人服务器上私自停止桌面服务、杀其他人的进程、锁 GPU 时钟或更换驱动。

**共享 GPU 被别人的计算任务占用时，停止正式性能测量**；搭建代码、编译、逻辑检查可以进行。记录当前计算进程及显存占用，不要用过去的 `who` 结果判断现在是否空闲。

### 2.2 初始化命令（可直接运行）

```bash
set -euo pipefail
mkdir -p "$HOME/fp4_bf16_study"/{third_party,build,src,include,config,scripts,raw,profiles,figures,reports,tests}
cd "$HOME/fp4_bf16_study"
{
  date -Is
  hostname
  uname -a
  nvidia-smi
  nvidia-smi -q
  command -v nvcc || true
  nvcc --version || true
  command -v ncu || true
  ncu --version || true
  command -v nsys || true
  nsys --version || true
  gcc --version | head -n 1
  cmake --version | head -n 1
  python3 --version
} > raw/env_first.txt 2>&1
ncu --list-sections  > raw/ncu_sections.txt 2>&1 || true
ncu --query-metrics > raw/ncu_metric_inventory.txt 2>&1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > raw/compute_apps.csv 2>&1 || true
```

必须从编译后的 CUDA 小程序打印 `cudaGetDeviceProperties`：name、`major.minor`、`multiProcessorCount`、`l2CacheSize`、`regsPerMultiprocessor`、`sharedMemPerMultiprocessor`、`maxThreadsPerMultiProcessor`、`warpSize`、`clockRate`、`memoryClockRate`（字段支持以当前 CUDA 版本为准）。**不要用论文中 H100 的 64 warp/SM 或网上的缓存容量代替这台机器实际属性**。

### 2.3 硬件计数器权限的实际验证

仅 `ncu --query-metrics` 正常还不能证明有计数器权限。先编译 E0 的 smoke test，再运行：

```bash
ncu --set basic --launch-count 1 \
  -o profiles/smoke_basic \
  ./build/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm \
  --m=128 --n=512 --k=4096 --iterations=1
```

执行路径以**实际 CMake 输出目录**为准。若 NCU 报 `ERR_NVGPUCTRPERM`，请管理员为计数器授权，不得在无权限情况下臆造百分比；若在共享 GPU 上 NCU 被其他进程抢占，也先协调独占测量窗口。

监控频率与温度：

```bash
nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu \
  --format=csv -lms 500 > raw/gpu_clock_trace.csv
# 用单独的 shell 启动，实验后 Ctrl-C；不要把监控程序的 CPU 消耗混入结果。
```

对所有 BF16/FP4 条件做随机区组（ABBA 或随机顺序）测量；保持相同工作温度区间。显著降频、超功率或其他用户进入时整组作废并记录。

---

## 3. E0：获得真正的原生 FP4 路径（不能跳过）

### 3.1 精度选择

- **主基线**：BF16 输入、BF16 权重、FP32 累加、BF16 输出；推荐基于 cuBLASLt 和 CUTLASS 两套 BF16 实现，对最好者作为“强 BF16 基线”。
- **目标**：**NVFP4 W4A4，E2M1 payload、每 16 个元素一个 E4M3 局部 scale、全局 FP32 scale（如所选量化方案采用），FP32 累加、BF16 输出**。
- **FP16**：只保留小型 sanity check，验证 BF16/FP16 不是主要混杂因素，不在第一阶段做全部参数扫描。
- **FP8/MXFP4**：不进入首轮主线；先完成严格的 BF16↔NVFP4 实验。以后再用于检验精度连续变化规律。

**需要特别注意**：NVFP4 格式的理论值 `x ≈ q_E2M1 × s_local_E4M3 × s_global_FP32`。常见 block16 的理想平均存储是每值 `4 + 8/16 = 4.5 bit`，不包含物理布局 padding、全局 Scale、对齐和额外缓存副本。对于同一值，不得随意将 `MXFP4` 的 E8M0/block32 scale 当作 NVFP4 的 E4M3/block16 scale。

### 3.2 CUTLASS 官方 79a：**仅用于原生执行通路 smoke test**

官方链接：

- `https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`
- `https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/CMakeLists.txt`

源码明确写明：`ArchTag = cutlass::arch::Sm120`、`OpClassBlockScaledTensorOp`、`ElementA/B=nv_float4_t<float_e2m1_t>`，实现使用 `mma.sync.aligned.block_scale`。BF16 输出、FP32 累加。79a 的标准 Tile 约为 `128×128×128`，但会随具体 CUTLASS 版本变化。**其 `initialize()` 随机生成 A、B、SFA、SFB**，没有把一份 BF16 原始张量真正执行定标量化；因此只证明原生 Kernel 能运行，不可拿示例随机结果跟真实 BF16 模型输出直接比较。

```bash
cd "$HOME/fp4_bf16_study"
git clone --recursive https://github.com/NVIDIA/cutlass.git third_party/cutlass
cd third_party/cutlass
git rev-parse HEAD | tee ../../raw/cutlass_commit.txt
cd ../..
cmake -S third_party/cutlass -B build/cutlass \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_ENABLE_TOOLS=OFF
cmake --build build/cutlass --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j 4
find build/cutlass -type f -name '79a_blackwell_geforce_nvfp4_bf16_gemm' -perm -111
```

> 这些 CMake 参数是**起点**，实际依赖最新 CUTLASS 版本及 GPU/编译器环境；如果配置失败，保存完整报错并调整，**不可悄悄退回别的架构或纯 BF16 路径**。官方例子要求 CUDA Toolkit >= 12.8。特别注意此示例在工具链/架构不支持时可能打印提示、直接 `return 0`；退出码为 0 **不等于跑了 GEMM**。必须看到通过验证、真实性能报告，且 Nsys 显示设备 Kernel 启动。

### 3.3 从 PTX 与 SASS 双重确认

编译时保留 LineInfo (`-lineinfo`) 及合适的 `-O3` 配置，并导出：

```bash
# BIN 由上一步 find 的结果填写，不在这里猜测路径
cuobjdump --list-elf "$BIN" > raw/fatbin_elf.txt
cuobjdump --dump-sass "$BIN" > raw/nvfp4_sass.txt
cuobjdump --dump-ptx  "$BIN" > raw/nvfp4_ptx.txt || true
grep -Ei 'MMA|OMMA|BLOCK_SCALE' raw/nvfp4_sass.txt | head -n 60
```

**验收不是要求 SASS 必须出现某个唯一字符串**：PTX 语义应匹配 `mma.sync.aligned.block_scale`；实际 SASS 的 mnemonic 可能随工具链变化，必须从 SASS 和 profiler 两方面核对 FP4 Block-Scaled Tensor MMA 是否发生。不要将 `tcgen05`、`wgmma`、普通 BF16 MMA 或“先反量化再 BF16 GEMM”称为这里的 SM120 原生 NVFP4 路径。

### 3.4 同一份源数据怎样生成

必须实现两套**独立且可核查**的数据分布：

**D0：硬件指令公平性数据**——生成可被 BF16 和 NVFP4 精确或近精确表示的非零值（例如由 E2M1 值乘有界幂次 scale），使两路径在数值上尽量相同。作用是最大限度排除数值误差和异常值，同时测指令/数据供给差异。不能在测试中让很多元素都恰好为 0 或大量整块 scale 均为 1，却声称覆盖真实量化情形。

**D1：量化真实开销数据**——从同一 BF16 原始张量中量化 A、B：

1. 用固定 seed 生成 `A_bf16[M,K]`、`B_bf16[K,N]`，包括正态分布、长尾分布、零稀疏及代表性真实模型分布；在每个尺寸固定源张量和布局。
2. 分别对 A、B 计算整体 `amax_global` 和每 16 元素分组 `amax_local`；按照选定的 NVFP4 recipe 算 FP32 global scale、E4M3 local scale；按 E2M1 最近舍入/饱和策略量化 payload。
3. 针对选定的 rowwise/columnwise operand 生成**与 CUTLASS SFA/SFB 约定匹配的 scale swizzle**；记录逻辑形状、物理 shape、bytes、padding 和内存布局。**禁止**仅把 `[M,K/16]` 的连续 Scale 数组传给一个需要交错布局的算子。
4. 定义 `A_q_dequant`, `B_q_dequant`：利用**实际存储后的 E2M1、E4M3、FP32 global scale**反量化。确保原生 CUTLASS Kernel 与数值参考施加的是同样的 global scale 语义；若示例只接受 block scales，则通过精确、可审计的 epilogue 或数据路径整合 global scale，不要漏乘或重复乘。
5. BF16 和 NVFP4 输出统一 BF16。准确性评价依赖上面两个参考，不可直接用误差很大的原始 BF16 对照判定 FP4 Kernel 算错。

**正确性“双金标准”**：

```
Ref_original = FP32_GEMM(float32(A_bf16), float32(B_bf16))
Ref_quant    = FP32_GEMM(float32(A_q_dequant), float32(B_q_dequant))

BF16_implementation ── 对 Ref_original 测数值实现误差
NVFP4_native          ── 对 Ref_quant    测数值实现误差
NVFP4_native          ── 对 Ref_original 测量化额外误差
```

输出 `max_abs_error`, `RMSE`, `relative_L2`（分母接近 0 要保护）, cosine, P99 absolute error, NaN/Inf count；误差容忍度由**同尺寸同精度的 CUTLASS 参考及可观察的 FP32 累加舍入**标定，不要随意写一个绝对阈值。小尺寸 test 至少包含 `M=16/128，K=128/4096，N=128/512`，并对 scale/padding/nonzero 输入做单元测试。

### 3.5 G0–G6：真正可解释的对照矩阵

| ID | 数据在 GEMM 前的状态 | 乘法执行方式 | 目的/限制 |
|---|---|---|---|
| **G0** | 原始 BF16 A、BF16 B | BF16 Tensor Core | 未量化主基线 |
| **G1** | 量化再反量化的 A/B，**预先存 BF16** | BF16 Tensor Core | 和 G3 使用尽量相同数值（不是存储公平），隔离数值/输出因素 |
| **G2** | NVFP4 A/B + scales | 在 Kernel 内反量化至 BF16，再 BF16 MMA（若可高效构造） | 同样压缩输入，考察解码+BF16 计算成本；若实现难以公平比较须标注“研究性实现” |
| **G3** | 同一份 NVFP4 A/B + scales | SM120 原生 Block-Scaled MMA | 目标原生路径；G1 vs G3 仍包含存储/实现差异；G2 vs G3 包含解码/数据搬运差异 |
| **G4** | NVFP4 权重、BF16 激活 | Weight-only on-the-fly dequant 后 BF16 MMA | 观测仅压缩权重的收益；**W4A16 不等于 W4A4** |
| **G5** | 原始 BF16 + 离线量化好的 NVFP4 权重、在线量化激活 | 两条**完整**独立前向路径 | 包含在线激活 quant/Scale/format/launch、GEMM、epilogue，与真实推理更相关 |
| **G6** | 优秀的 AWQ/INT4+Marlin 或官方库实现（可用时） | 真实生产路径 | 工程强基线；具体运算路线必须审计，不得称 INT4 一定是原生 INT4 Tensor MMA |

**重要论断**：没有任意两组可以被简单宣称为“纯数据类型效果”。这一矩阵提供的是**多方交叉证据**；只有结合指令微基准、SASS/NCU 和相同实现家族的配置控制才能做谨慎归因。

### 3.6 G5 在线/离线量化成本规定

权重在部署前可以离线量化并放 GPU，**core** 测量只对预先 pack 好的权重和激活做 GEMM。**full** 从 BF16 激活开始，实际执行在线激活量化 + 必要 Scale、swizzle/packing、GEMM、输出，且权重仍可离线量化。必须报告每阶段 GPU 执行时间与总时延（禁止把动态量化成本放到 CPU 计时外）。若量化直接融合在前一层 epilogue，单独报告 fused producer 的全路径成本，并与同一融合等级的 BF16 强基线比较。

权重 offline 的含义是“部署预处理”而不是“必须每 decode step 重新量化”；不要故意在每次 step 对完整权重量化，从而人为抬高 FP4 成本。

---

## 4. E1：单 Kernel 性能与硬件资源画像

### 4.1 负载清单及顺序

**第一梯队：** 固定 `K=4096, N=12288`，`M=16,32,64,128,256,512,1024`。这是 FFN 类逻辑 GEMM 形状，但并非声称所有具体模型都采用这一尺寸。

**补充 Decode：** `M=1,2,4,8`；注意主 GEMM 79a 未必适合极小 M，需选择支持并高效的生产 Kernel，记录 unsupported shape 和 padding，不能为跑通强行做大量无效计算却隐瞒。

**L2 容量实验：** 固定 K=4096，M=128/512，N=4096、16384、65536（记录具体支持/内存占用），保证有：两格式权重均可能放入 L2、FP4 更可能放入而 BF16 不易放入、两者都大于 L2 的三个阶段。对于这个 K，单份权重理论字节（忽略 Scale 物理 padding）：

| N | BF16 权重 | NVFP4 权重+局部 scale（理论） |
|---|---:|---:|
| 4096 | 32 MiB | 9 MiB |
| 12288 | 96 MiB | 27 MiB |
| 16384 | 128 MiB | 36 MiB |
| 65536 | 512 MiB | 144 MiB |

5090 的 L2 公开规格常见为 **96 MiB**，但最终以 `cudaDeviceProp.l2CacheSize` 与 NCU 实测为准；**`N=12288` 的 BF16 权重约等于名义 L2，不能武断地归入“完全 L2 命中”或“完全 L2 放不下”。** 输入、输出、其它工作集、替换策略、set-aside 策略、padding 都会影响实际命中。

### 4.2 四类测量必须分别运行

1. `core_warm`：A/B 已按所需格式在 GPU，重复读同一权重/激活；先预热，正常执行采样。记录真实 L2/DRAM 读取。
2. `core_cold`：由应用维护一个超过 L2 的驱逐缓冲区或轮转多份权重；实验前核对驱逐行为，**不能仅凭设置名字叫 cold**。
3. `full_warm`：在线量化/Scale/布局准备（需要时）+ GEMM + epilogue；保持与 core 相同权重缓存方案。
4. `full_cold`：完整路径在轮转权重/驱逐条件下运行，单独评估是否具有实际推理代表性。完整路径全程用 CUDA Events 或 graph 计时，不能把各子 Kernel 的独立测量直接加和当作真实系统时间。

**独立 timing 程序**：每样本先 CUDA sync 确定上一轮结束，随后在同一 stream 的 event 之间提交固定的 `K` 次 Kernel（或 K 次 Graph 节点），测 `T_sample / K`。建议每种组合 30–100 个**独立样本**×至少 3 个独立会话，K 使一个样本的总 GPU 时间至少 1–5 ms；对应的单 Kernel 时间由 `T_sample/K` 估计，明确缓存状态是否因 K 次重放改变。所有采样顺序交错；记录 CPU launch / Graph 版本和实际 GPU 时间。对 NCU 再单独抓一个典型 Kernel，不要把 NCU 的执行时间混入正常 timing 数据。

**吞吐**：`effective_TFLOPS = 2*M*N*K / time_seconds / 1e12`。这是逻辑 GEMM 有效吞吐，不能等同于**物理实际执行 FLOPs**（padding/split-K 可能做更多工作），也不能把 `TFLOPS/宣传峰值` 当作唯一资源占用指标。

### 4.3 NCU 分层采样，不要一口气收几百个指标

建议每个精度先测 M=16/128/512，再推广全 M。

**Tier 1：低开销概况**

```bash
# 下述 ./bin/gemm_bench 是由执行者实现的研究程序，不是 CUTLASS 自带命令
ncu --set basic --launch-count 1 --cache-control none \
  --nvtx --nvtx-include "E1_MAIN_G3_M128/" \
  -o profiles/e1_g3_m128_basic \
  ./bin/gemm_bench --group G3 --m 128 --n 12288 --k 4096 \
    --data D1 --mode core --cache warm --profile-once
```

**Tier 2：资源分解（以本机存在的 section 为准）**

```bash
ncu --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --cache-control none --replay-mode application \
    --nvtx --nvtx-include "E1_MAIN_G3_M128/" \
    --launch-count 1 \
    -o profiles/e1_g3_m128_detail \
    ./bin/gemm_bench --group G3 --m 128 --n 12288 --k 4096 \
      --data D1 --mode core --cache warm --profile-once
```

> 某些 section 在具体 NCU 版本的准确名称或支持状态不同，按 `ncu --list-sections` 过滤；必须保存原始错误和实际最终使用的指令。**上述 NVTX 过滤要求程序用 `nvtxRangePushA("E1_MAIN_G3_M128")` / `nvtxRangePop()` 将所需 GEMM Kernel 包起来；若程序没有实现 NVTX，这两个命令不会自动抓到目标 Kernel，应先补齐而不是删掉过滤盲抓。Kernel Replay 默认会对采集内核进行序列化与默认缓存清理；Application Replay 可与 `--cache-control none` 配合，让应用控制缓存。** 这不是“只要加这两个参数热缓存就一定成立”，必须由 DRAM/L2 计数核验。

建议存储本机真实 metric 名称清单，并用 `ncu --query-metrics` 逐项查找：

- 计时/时钟：`gpu__time_duration.*`、`sm__cycles_elapsed.*`、频率与限频原因。
- Tensor 运算：以 `sm__pipe_tensor...`、`smsp__inst_executed_pipe_tensor...` 或当前产品实际提供的指标为候选；不能用另一个 GPU 的指标名强制执行。
- FP32：`sm__pipe_fma...` / 实际 FP32 pipe 指标和 SASS 指令分类。
- 访存：`dram__bytes_read.sum` / `dram__bytes_write.sum` 或当前 NCU 等价指标、`dram__throughput...`、`lts__...` sectors、L1/L2 命中与吞吐。
- Warp：`smsp__issue_active...`, `smsp__warps_eligible...`, `smsp__warps_active...`，以及各类 `smsp__warp_issue_stalled_*`。
- 静态资源：register/thread、sharedMem/CTA、waves/SM、theoretical/achieved occupancy、local memory spilling。

**metric 语义要求**：

- `Tensor Pipe Active` 是流水线活跃比例，非实际 FP4 FLOPS/理论峰值、也不等于其余周期必能执行一段独立任务。
- `dram throughput %` 应结合 DRAM 读取的**字节数**、时间、实际可达到的 STREAM 峰值、缓存模式一起解释。
- L2 hits 是实际 transactions/sectors 的统计，不能只用 `N×K×dtype bytes` 推断实际流量。
- 高 Occupancy 不一定好；低 Occupancy 不必然坏；Eligible Warp 低需结合 Warp Stall 类别解释。
- 多数**普通 Kernel Replay 的单 Kernel 指标不能证明与另一个 Kernel 真正同时驻留**。

### 4.4 E1 CSV 必需列

`timestamp,git_commit,cutlass_commit,driver,cuda,ncu,device_uuid,M,N,K,group,data_id,quant_recipe,src_hash,kernel_name,kernel_backend,layoutA,layoutB,tileM,tileN,tileK,stages,splitK,smem_bytes,registers_per_thread,spills,workspace_bytes,payload_bytes,scale_bytes,padded_bytes,mode,cache_state,mem_clock,sm_clock,power_w,temp_c,time_us_p10,p50,p90,mean,std,tflops_effective,dram_bytes_read,dram_bytes_write,l2_bytes_read,l2_hit_rate,tensor_pipe,fp32_pipe,issue_active,eligible_warps,stall_top3,device_busy_other_user`。

缺失字段填 `NA` 且说明原因；保存 NCU 完整导出与原始 timing JSON，不准填估计值当实测。

---

## 5. E2：从“指标差异”走向“瓶颈归因”

### 5.1 E2-A：计算单元极限/受控压力曲线

**问题**：在既定 M/K/N 下，BF16/NVFP4 是否真受 Tensor Core 吞吐限制？或者是数据供给、Issue 或依赖限制？

**两组不同用途的实验，不能混用：**

- `E2A-peak`：独立微基准，BF16 MMA 与 Block-Scaled NVFP4 MMA 分别使用片上复用的操作数、多个独立 accumulator，扫 ILP=1/2/4/8 与 repeat=1/2/4/8。通过编译期固定展开/运行时写出 checksum 保证操作不被消除；对比 SASS 总 MMA 数量。先验证该微基准是否主要受 Tensor Pipe 而不是寄存器/Issue 限制。此测试用于**建立 SM120 实测 compute ceiling**，不是生产 GEMM 性能。
- `E2A-main`：生产 GEMM 保持逻辑运算不变，改变 Tile、Stage、K 分块、调度策略（先只用合法候选），记录时间/指令数/资源/访存。不得为了制造“余量”强制往生产 GEMM 增加一批没有语义的 MMA 后声称是同一应用。

**必要对照**：

1. 同一个 native FP4 微基准，对比 `tensor MMA repeat` 增长的时间斜率与真正发出的 MMA 数量。
2. 独立 FP32 FMA 微基准形成非 Tensor 计算上限。
3. 如果 FP4 大量额外 MMA 代价很低，检查是否单纯因为复用了 register operands、原运行尚未填满数据供给；**只有后面的双任务 E3 能检验这是否可供独立有用任务使用**。
4. 不要求 BF16/NVFP4 的指令数相同；要求相同逻辑工作量与明确的物理指令计数。

### 5.2 E2-B：DRAM、L2、SMEM 与数据供给敏感性

1. 用大数组 `stream_copy` 测实测 DRAM 可达带宽（工作集至少数倍于 L2，尽可能避开 L2 命中），分别读/写/Copy 模式，记录是否受写回、ECC/其他流量、功率与频率影响。
2. 用可重复读取的中小数组测 L2 读吞吐，不要只用 GPU 芯片公布的 L2 峰值。
3. 对 GEMM，固定 (M,K,N)，比较 warm、evicted、rotation；记录 `T, dram_read_bytes, L2 hit, L2 sectors, sector miss, memory stalls`。
4. 控制权重容量：`N=4096/16384/65536`。这组特别能识别 **“FP4 仅仅因为压缩后更容易驻留 L2”** 的收益，而不是 Tensor Core 指令优势。
5. 单独量化 A（在线）、生成/重排 Scale、packing、权重重新布局，逐段报告时间及 bytes；如果它们已与 producer/epilogue 融合，按真实执行图记录不应强行拆出后相加。
6. `L2 cache window` 可作为辅助实验，使用 CUDA 适当的 access policy window 需记录可用性；不要为了缓存控制改变生产 Kernel 后把两者结果直接当作相同配置。

**诊断方向举例（不是结论）：**

- Warm 快、Cold 慢且 DRAM bytes 同步变大 → 说明对缓存驻留敏感。
- FP4 Tensor Pipe 低、Issue 接近饱和、Scale/layout 指令多 → 可能主要被指令供给/数据组织限制，后续用 E3 的 issue 探针验证。
- 改 Tile/Stage 让 Tensor Pipe 提高但 kernel 变慢 → 不能单看利用率，应检查访存、Occupancy、同步与 epilogue。

### 5.3 E2-C：寄存器、Shared Memory、Issue 的资源约束

每个精度选择**各自最优** Kernel，再编译至少 2 个次优版本（合法 Tile/Stage）：

- `fast_alone`：独立最好。
- `smem_light`：降低 SMEM 或数据驻留 footprint，可能降低主 Kernel 性能。
- `reg_light`：降低每 CTA 寄存器量；注意编译后 spilling。

每版本记录：静态寄存器、实际本机可驻留 CTA 数、动态 shared、local memory loads/stores、执行时间。切勿随意设置 `-maxrregcount` 后不检查 spilling，就声称产生更多资源。

建立可驻留性预算：

```
regs_A_CTA = registers_per_thread_A × threads_per_CTA_A (含分配粒度取整)
regs_B_CTA = registers_per_thread_B × threads_per_CTA_B (含分配粒度取整)
SMEM_A + SMEM_B <= physical_limit_by_SM
regs_A + regs_B <= physical_limit_by_SM
warp_A + warp_B <= warp_limit_by_SM
block_A + block_B <= CTA_limit_by_SM
```

这是**必要不充分条件**：硬件调度、warp scheduler、寄存器分配粒度和其他约束会进一步影响实际 co-residency。不同 Kernel 可分散到不同 SM，因此“不能同 SM”不等于“不能在 GPU 层面并发”。

---

## 6. E3：不同精度下的双任务干扰矩阵（整项研究的核心）

### 6.1 主任务与探针定义

主任务为 G0 或 G3（另可选 G5 full），先选 `M=16, 128, 512; K=4096, N=12288`。最初选 G0/G3 是为了让主任务 kernel 的 core 资源竞争足够清晰；完整路径另在 E4 评估。

至少设计 5 类探针：

| 探针 ID | 实现方式（可调参数） | 预期主要施压资源 | 主要误判风险 |
|---|---|---|---|
| P1-FP32 | 线程内多独立 FP32 FMA 链，控制 ILP、CTA 数、循环次数、活跃 warp | FP32 pipe / issue | 寄存器增加也可能阻止 co-residency；不保证只占 FP32 |
| P2-TENSOR | 官方合法 BF16/NVFP4 MMA，自复用数据，控制 K 重复和 CTA 数 | Tensor pipe / issue / operand path | 可能同时抢寄存器、SMEM 与 issue |
| P3-SMEM | 按 thread/block 使用 shared memory，控制地址步长、访问次数、读写比例、bank conflicts | SMEM/L1 data path | bank conflict 使 warp stall/issue 模式改变 |
| P4-L2 | 工作集小于实测可用 L2，反复读随机/固定地址 | L2 带宽/缓存替换 | 也需 DRAM 填充；地址模式影响 cache miss |
| P5-DRAM | 大于 L2 多倍的数组流式读写，调节 Grid 与字节数 | DRAM 带宽 | 即使读写逻辑字节数大，也可能复用 L2；用 counters 验证 |
| P6-ISSUE（条件扩展） | 低内存需求、多独立指令、调节每 Warp 发射压力 | issue slot | 同时占用其它 pipe，需要多证据归因 |

每一类探针的**单独校准曲线**必须完整保存。压力分三档 low/mid/high，但这些档位要由**单独测试的实际吞吐、功率、issue/pipe/byte counters**确定，不能简单令 `CTA=1/2/3` 就叫 25/50/75% 压力。

### 6.2 核心计时协议（重要）

四个基本实验：

1. **A-alone**：主 GEMM（或其固定次数重复）独立，记录独立 `time_A`, `completed_A`, `occupancy_A`, GPU 频率。
2. **B-alone**：探针独立，记录 `time_B`, `completed_B`, `bytes_B`, `freq_B`。
3. **Serial A→B**：同一个进程、同一缓存/时钟实验政策，依次执行完整固定工作量，记录 `T_serial`。
4. **Concurrent A||B**：同一进程两个非默认 CUDA Stream，使用一个 ready event 触发开始，分别记录结束事件，控制 Stream wait 两个结束事件得到 `T_joint`。Graph 可用于减少 CPU 端 launch 间隙，但要核对 Graph 是否改变并发能力；不把 GPU 调度启动顺序等同真正同时执行。

**为严谨解释“共执行”**，必须有 NSYS 时间线证明 A、B 在 GPU 上真的有重叠；若只能并发提交而无重叠，应归为调度/驻留障碍，不应计算为资源争抢的 slowdown。

**并发完成时间的合理定义**：用统一的 `ready` 标记起点，单独控制 stream `wait(stopA)`, `wait(stopB)` 后记录 `joint_stop`；事件测量 A/B 延迟应在各自 stream 的 start/end 之间；详细取值以 GPU 事件语义与 Nsight Systems 时间线对齐。注意同步本身不能使用 CPU 逐个串行 launch 后再启动另一条 stream 的做法。

**短 Kernel 处理**：主 Kernel 如只有几微秒，CPU launch 噪声可能主导。可以为每个任务建立独立重复的 CUDA Graph，让一个联合样本持续 10–50 ms；同时必须确认**节点级时间交叠，而不仅仅是两个 Graph 的墙钟窗口重叠**。对窗口内每个任务的真实工作量和完成次数分别计数。

### 6.3 五个必须报告的指标

令 `T_A^alone, T_B^alone, T_A^conc, T_B^conc, T_joint, T_serial` 为正常非 NCU 条件下的 GPU 事件时间：

```
S_A = T_A^conc / T_A^alone       # 主任务减速
S_B = T_B^conc / T_B^alone       # 探针减速
G   = T_serial / T_joint        # 联合加速；>1 才说明并发对总完成时间有益
R   = S_A_FP4 / S_A_BF16         # 同干扰强度的相对敏感性（只描述现象）
Throughput_A = completed_A / T_joint
Throughput_B = completed_B / T_joint
```

必须画出横轴 A Throughput 或主任务 slowdown、纵轴 B Throughput 的 Pareto 曲线。**只看 G 或 R 不够**：两个精度 A 独立执行时间差很多，同一 B 可能导致不同的暴露时间；要求一个对照为**相同探针总工作量**，另一个对照为**相近探针持续时长/相同时间窗口**。必要时另设固定总工作量/固定总体占用约束。

**严禁使用**`(T_A+T_B)/T_joint` 代替两项实际工作量的等价基线；必须使用实验测量的 `T_serial`。有缓存/温度/启动副作用时两者不必相等。

### 6.4 NCU/Nsys 分工——与 V1 重要区别

- **普通 NCU Kernel Replay**：用来 profile 单独的 A 或 B，审计各自资源负载；**不能直接用于确认 A||B 的并发时间**。
- **Nsight Systems**：观测真实 CUDA Stream、Kernel 开始/结束、NVTX、Graph 节点时间线；这是识别是否重叠的第一工具。
- **必要时 Nsight Compute `Range Replay` 或 `Application Range Replay`**：官方文档明确支持在一个 range 内测量必须并发的 kernels，但指标通常归于**整个范围**，并有 replay 的一致性/支持限制。因此它用于聚合瓶颈佐证，**不能再把区间计数简单分摊给两个 Kernel**。
- **正常 CUDA Events/Graph 执行**：主时间结论和加速来自这里，不拿 profiler 下的时间作为最终性能数字。

基本 NSYS 命令（以已实现程序接口为准）：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  -o profiles/e3_g3_m128_fp32_mid \
  ./bin/co_run_bench --group G3 --m 128 --n 12288 --k 4096 \
    --probe fp32 --pressure mid --mode concurrent --repeats 50
```

利用 NVTX 标识任务窗口和 Kernel 组。用 `nsys stats` 或 GUI 审核每个任务是否真 overlap（包括 overlapping fraction、kernel end、空窗、Context Switch）。若 NSYS 没有足够明确的 same-SM 共驻证据，只写**同 GPU 时间重叠**，不得写**同一 SM 共驻**。

### 6.5 最小扫描矩阵（不要一开始爆炸搜索）

**探索轮**：`3 M × 2 precision × 2 probe（P1/P5）× 3 pressure = 36` 个条件；每条件分别测 alone、serial、concurrent，3 个独立会话；发现可疑差异后扩展 P2、P3、P4。

**消融**：

- CUDA stream / launch sequence / graph yes-no。
- 探针 CTA 资源：reg/smem 更小的候选，使本来不能驻留的任务变得可能驻留，确认原因。
- BF16 和 FP4 分别采用各自 `fast_alone` 和 `co_run_friendly` 两类 Kernel。
- 同一工作量下调节 main 与 probe 的**计算相对规模**和负载持续时间，排除短任务随机重叠。

**首轮积极结果的复核门槛**：重复会话 3 次以上、随机交错、95% CI 排除无差异，并且差异大于实际运行抖动；5% 变化可作为复核触发，不是自然科学上的统计显著阈值。

---

## 7. E4：从“资源差异”到真实有用任务收益

### 7.1 FFN gate/up（合法独立分支）

```
G = X @ W_gate
U = X @ W_up
Z = SiLU(G) * U
Y = Z @ W_down
```

测试以下 4 类强弱层次：

- F0：分别串行执行 gate、up、SiLU×、down。
- F1：两个 Stream `gate || up`，正确 event dependency 后 SiLU×、down。
- F2：**合并 gate/up 投影**（一次 `X @ [W_gate, W_up]` + 对应后处理），如现有框架已有合并路径必须加入。
- F3：现成优化 fused FFN 或其他 CUTLASS/生产框架已有高性能路径，能用则纳入。

每一个 F0–F3 **各自在 BF16 和 NVFP4 的 full 路径**运行；FP4 同一 `X` 的动态量化可否在 gate/up 两支共享必须同等考察。其他阶段的依赖严禁改变：down 不能在 Z 就绪前开始；也不能因为 G/U 合并就偷偷改变数值目标或输出 dtype。若 fused gate/up 始终最好，`gate||up` 加速不能被包装成研究贡献。

### 7.2 更通用的独立“准备工作”

只允许**语义上已就绪的**工作：

- 某个独立请求下一步已知的权重/布局 metadata 或预处理；
- 当前算子之外、已经由前驱生产好的 Tensor 做 quant/repack；
- 明确已知地址、可被预取且可能提高后续命中率的数据（同时评估 DRAM 带宽/缓存污染成本）。

**不能**提前量化尚未算出的下一层激活，也不能把纯粹无用 FP32 循环的 overlap 当成推理性能提升。

对每一候选必须同时验证：

1. 不影响主任务与辅助任务最终值（精度允许误差需同量化配置）。
2. 附加 GPU 字节流量、Tensor/FP32 指令、SMEM/regs/指令发射是否导致主任务减速。
3. 当前任务 + 后续消费者的**完整 makespan 是否变短**。
4. 辅助工作在何时已经依赖就绪，在何时被消费者使用。
5. 对比同精度下的 best serial、best fused、best concurrent，而非仅相对 BF16 baseline。

### 7.3 若 E3 发现的瓶颈不能被 E4 利用

明确报告 no-go：差异只发生在 synthetic probe、独立任务无合法并发窗口、寄存器/SMEM 阻止共驻、交错引起 L2/DRAM 竞争、精度误差超过模型质量约束、最佳融合路径已消除收益等。**不因为缺少正结果强行切换到 FP4 KV Cache 压缩。**

---

## 8. 统计学、功率和可重复性细则

- **实验会话**：同一进程完整测试一个 block 的配置为 session；跨不同时间重复 3–5 session；每 session 30–100 个事件样本，每个样本内部 10–1000 次 repeat 不当成独立实验单位。
- **随机化**：同一 M 的 G0/G3 交错，low/mid/high 也交错；不得 BF16 全测完才测 FP4，防止温升与 DVFS 偏差。
- **时钟**：不经管理员许可不修改 GPU 时钟；记录实际 `sm_clock`, `mem_clock`, power, temp, PerfCap；如时钟漂移与差异同数量级则结果待验证。
- **多比较**：初筛 36 组合时发现某一点显著，不应只展示最佳一例。报告所有组合、置信区间和反例，强效应另做保留集重复。
- **置信区间**：优先对 session 层/区组内比值进行 cluster bootstrap 或以 session 中位数报告区间，避免将单 Graph 内千次重复视为千个统计独立观测。
- **正确性**：每个 Kernel 变体均验证；特别检查 FP4 packed bits、Scale 向量布局、全局 scale、输出 buffer、C/D alias、alpha/beta、padding、split-K、NaN。
- **负载来源**：先 synthetic 构造机制证据，再采集真实 BF16 LLM 层（固定模型 revision、token/position、shape、激活值、权重 hash）；不要直接将微基准推广为整模型结论。
- **资源占用**：动态 `GPU-Util` 不等于 Tensor Core utilization；`Occupancy` 不等于 pipe active；有效 TFLOPS 不等于 peak pipe 指标；理论压缩率不等于实际 DRAM 字节比。

---

## 9. 可交付的程序接口规范（必须实现，不是现成命令）

**程序 A：`gemm_bench`**

```bash
./bin/gemm_bench \
  --group G0|G1|G2|G3|G4|G5 --data D0|D1 --seed 42 \
  --m 128 --n 12288 --k 4096 \
  --backend cutlass|cublaslt|marlin \
  --mode core|full --cache warm|evict|rotate \
  --sample-count 50 --graph-repeats 100 \
  --profile-once 0|1 --json raw/e1_g3_m128.json
```

- 打印并写 JSON：程序是否真的运行了目标 GEMM，native/mma 检验路径；如果 G2 当前无法实现，输出 `UNSUPPORTED` 而不是偷偷换成 G1/G3。
- 依照 `profile-once` 在 GEMM 正式调用前预热，在核心区域标注 NVTX `E1_MAIN_G3_M128`；确保 NCU 不会误抓 quantize/host reference。
- `profile-once=1` 应只发出用于抓取的 kernel 数量（避免抓到数百个重复 kernel）。
- `--cache=evict` 必须记录驱逐策略、工作集大小、预热/驱逐后的 L2/DRAM 指标。
- `--mode=full` 必须允许单步拆分 time：`quantA`, `scaleA`, `swizzleA`, `gemm`, `epilogue`, `total`，并验证 total 与完整执行时间一致（融合实现不得强行虚构阶段耗时）。

**程序 B：`micro_probe`**

```bash
./bin/micro_probe --kind fp32|mma_bf16|mma_nvfp4|smem|l2|dram|issue \
  --ilp 4 --iters 4096 --ctas 170 --threads 128 \
  --bytes 268435456 --stride 1 --json raw/probe.json
```

- 每种探针的操作数、规模、依赖、寄存器/SMEM 使用、写出结果的防优化策略、实际资源占用，必须一起在日志中保存。
- `--kind` 不能默默改变其他硬件资源；若改变则在报告中解释混杂。

**程序 C：`co_run_bench`**

```bash
./bin/co_run_bench --group G3 --m 128 --n 12288 --k 4096 \
  --probe fp32 --pressure mid --mode alone_A|alone_B|serial|concurrent \
  --repeat-workload 256 --sample-count 50 --json raw/e3_one.json
```

- 同一份 A/B work definition 被四种 mode 复用，不能在 mode=concurrent 里改小任务量或数据集。
- 支持 NVTX 标注 `A_GEMM`, `B_PROBE`, `JOINT_WINDOW`；用 NSYS 审核真实 Kernel 交叠。
- 记录 A/B 实际完成数量与 `T_A`, `T_B`, `T_joint`, `T_serial` 及其统计样本，不只输出一个 speedup。

### 9.1 事件计时核心伪代码（由执行者实现并检查返回值）

```cpp
// 假定 A/B 的输入、GPU buffer、workspace 都已预先准备，避免 malloc/init 干扰。
cudaStream_t sa, sb, sc;
cudaEvent_t ready, a_start, a_end, b_start, b_end, joint_end;
// ... 创建非默认 streams 和 CUDA events，禁止 cudaEventDisableTiming 用于计时事件。

// 将 ready 放在控制流上，使两条计算流共享一个依赖起点。
cudaEventRecord(ready, sc);
cudaStreamWaitEvent(sa, ready, 0);
cudaStreamWaitEvent(sb, ready, 0);

cudaEventRecord(a_start, sa);
launch_A_same_workload(sa);
cudaEventRecord(a_end, sa);

cudaEventRecord(b_start, sb);
launch_B_same_workload(sb);
cudaEventRecord(b_end, sb);

cudaStreamWaitEvent(sc, a_end, 0);
cudaStreamWaitEvent(sc, b_end, 0);
cudaEventRecord(joint_end, sc);
cudaEventSynchronize(joint_end);

// T_A 用 a_start→a_end；T_B 用 b_start→b_end；
// T_joint 应使用 ready→joint_end 的GPU时间；
// 两条流的开始绝不保证刚好同周期，必须审核 NSYS timeline。
// 事件可在Graph capture兼容的正确位置安排；不要错误地在 graph 之外包一层
// host loop，然后假设 Graph 里不同流的工作自然能 overlap。
```

正式代码需要 `CUDA_CHECK` 检查所有 API，复用 events/streams，确保 `ready` 先有定义且没有对默认 stream 的隐性全局同步；同步/事件机制可能产生轻微启动成本，串行/并发基线都应一致控制。

### 9.2 原始 JSON schema 最低要求

```json
{
  "date": "2026-11-14",
  "gpu": "RTX 5090",
  "sm": 120,
  "run_id": "unique-run-id",
  "dataset": "D1",
  "group": "G3",
  "M": 128,
  "N": 12288,
  "K": 4096,
  "kernel": "exact-symbol-name",
  "native_nvfp4_verified": true,
  "quant_recipe": "NVFP4-1D-block16-E4M3-global-FP32",
  "mode": "core",
  "cache": "warm",
  "latency_samples_us": [],
  "mean_us": null,
  "p50_us": null,
  "p90_us": null,
  "dram_read_bytes_measured": null,
  "l2_hit_measured": null,
  "metrics_raw": {},
  "validity": {
    "correctness_passed": true,
    "concurrent_other_users": false,
    "nsys_overlap_verified": null,
    "notes": "No made-up measurements; null means unmeasured."
  }
}
```

**说明**：这是 schema 例子，不是测量结果；上面的 `native_nvfp4_verified=true` **仅允许 E0 的 PTX/SASS/NCU 证据通过后才填写**，不准按 group 名称自动填 true。

---

## 10. 分阶段验收清单与明确的停止条件

### M0：工具/格式/指令验收

- [ ] 记录机器属性、驱动、Toolkit、NCU/NSYS 版本、管理员允许的性能计数器权限和共享 GPU 状态。
- [ ] CUTLASS 79a 编译并运行，有 **实际 Kernel**、`Disposition: Passed`；不是空返回。
- [ ] PTX/SASS/NCU 中有对应的 SM120 Block-Scaled FP4 MMA 证据。
- [ ] 根据同一 BF16 源张量生成合法 NVFP4，并通过 decode→reference 对照测试。
- [ ] 输出了当前真实 L2 Cache、register/SM、SMEM/SM、warp 上限、工作时钟记录。

### M1：资源画像验收

- [ ] 同源数据下 G0、G1、G3、G5 至少可测（G2/G4 如实现困难要记录缺失/替代设计及科研风险）。
- [ ] `M=16,32,64,128,256,512,1024` 的完整 `core_warm`、`core_cold`、`full_warm` 表（不支持的 shape 明确标注）。
- [ ] 计算、访存、warp issue、Occupancy、regs/SMEM/CTA 等均有可核查的实测或明确缺失。
- [ ] 给出不少于 2 个、最多 3 个具体瓶颈假设；每个附支持、反例、下轮反证条件。

### M2：因果与干扰验收

- [ ] P1/P5 的独立压力标定、E2 计算与缓存敏感性完成。
- [ ] 首轮 36 组合，均有 alone/serial/concurrent 四类实验、独立 session/CI。
- [ ] 实际 overlap 使用 NSYS 验证；仅在存在 overlap 时分析同时资源竞争。
- [ ] 清晰标出 `fast_alone` 和 `co_run_friendly` 的寄存器/SMEM 资源差异。
- [ ] 任何正结论必须提供现象、硬件指标、反证消融三个层次的证据。

### M3：有用工作验收

- [ ] FFN gate/up 或另一合法依赖路径至少一项完整实现。
- [ ] BF16 和 NVFP4 在各自最强 baseline 下比较，在线激活量化/Scale 成本不遗漏。
- [ ] 真实 end-to-end 子图 `makespan`、质量/数值保持、内存/功率消耗、附加收益与失败案例都给出。
- [ ] 若无法稳定超越强 baseline，提交 no-go，不把 synthetic overlap 宣称为模型加速。

**强制报告 STOP/NO-GO 的触发**：找不到原生 MMA；量化数值/Scale 不能核查；NCU 没权限且用户需要硬件归因；多用户 GPU 无可重复独占测量；所谓优势只在 NCU Replay 下出现；只有 synthetic 额外 MMA 低成本而真实辅助任务总耗时不下降。

---

## 11. 工程目录结构和 Codex 工作纪律

```
fp4_bf16_study/
├── README.md
├── third_party/cutlass/             # 固定 git commit；不要直接魔改官方源
├── config/experiment.yml            # 固定维度、量化 recipe、cache、seed
├── src/
│   ├── device_info.cu              # cudaGetDeviceProperties + 环境
│   ├── nvfp4_quant.cu/.cpp         # 同源 BF16 → FP4 + scale + swizzle
│   ├── gemm_bench.cu               # G0/G1/G2/G3/G4/G5
│   ├── probe_fp32.cu
│   ├── probe_tensor.cu
│   ├── probe_smem.cu
│   ├── probe_l2_dram.cu
│   └── co_run_bench.cu              # 双流依赖 + GPU events
├── include/
├── tests/                          # scale/pack/precision/correctness
├── scripts/
│   ├── 00_env_check.sh
│   ├── 01_native_smoke.sh
│   ├── 02_baseline_timing.sh
│   ├── 03_ncu_profile.sh
│   ├── 04_sensitivity.sh
│   ├── 05_interference.sh
│   ├── 06_ffn.sh
│   └── plot_all.py
├── raw/ profiles/ figures/
└── reports/
    ├── M0_native.md
    ├── M1_resource_baseline.md
    ├── M2_interference.md
    ├── M3_application.md
    └── revision_log.md
```

**给执行者的规则**：

1. 先 M0，然后 M1；M1 未验收，不开发复杂 FP4 co-scheduler。
2. 编译失败必须保存 stderr、git commit、nvcc版本，不许悄悄替换实现；需要我或用户决策的选项，把可复现错误交回来。
3. 任何 GPU 测试先检查是否有其他计算任务；若非独占，只能做 correctness / smoke test 并标注“不可用于正式性能结论”，最好也避免对其他用户形成干扰。
4. 仅使用项目工作目录；禁止改系统驱动、内核模块、权限、别人的文件，或 kill Xorg/桌面进程。
5. 所有代码与图表标清数据来源、后端、量化格式和 `core/full`；禁止选择性汇报正结果。
6. 每个里程碑先交付**命令 + 原始文件 + 诊断 + 可以复现的结论**，然后才继续。

---

## 12. 建议首日只做的 6 件事（最小 go/no-go）

1. `nvidia-smi`、`device_info`、NCU permissions、`CUTLASS 120a` 依赖检查。
2. 编译官方 79a，运行 `M=128, N=512, K=4096` 及 `M=128,N=12288,K=4096`；确认逻辑正确并非空返回。
3. 用 `cuobjdump` + NCU 确认原生 FP4 block-scaled MMA；明确 CUTLASS 版本里的真实命名。
4. 实现**同源 BF16 → NVFP4** 量化并编译自定义 wrapper；先做 `M=16,N=128,K=128` 的单测，检查 local/global scales 的实际语义。
5. 选 `M=16,128,512; N=12288,K=4096` 做 G0/G1/G3 core_warm，且使用相同 BF16 输出、真实事件计时及 NCU Tier1。
6. 输出 `M0_native.md` 和第一版 `e1_pilot.csv`，用**数据和可复现命令**说明是否进入全量 E1。

除非上述步骤已有证据，不接受“FP4 Tensor Pipe 低，所以有空闲可拿来做 KV 量化”等结论。

---

## 13. 官方资料与版本检查清单

- CUTLASS SM120 NVFP4→BF16 79a 示例：<https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu>
- CUTLASS 79 系列 CMake targets（需要 `120a|120f|121a`）：<https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/CMakeLists.txt>
- CUTLASS 官方 Build：<https://github.com/NVIDIA/cutlass/blob/main/README.md>
- Transformer Engine NVFP4 数据格式：<https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html>
- Nsight Compute Profiling Guide（Kernel / Application / Range Replay，Cache Control）：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- Nsight Compute CLI（NVTX / metric 查询）：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/>
- Nsight Systems：<https://docs.nvidia.com/nsight-systems/UserGuide/>
- 之前的研究归档：`RTX5090_原生FP4大模型推理_完整探索记录_2026-11-13.md` 的 Round2 / Round4 / Round6 / Round7，用于了解已经发现的寄存器与并发问题。不得把此前 synthetic 实验宣传为新的完整模型实测。

---

## 14. 给执行者的验收回复模板

```
[Milestone] M0 / M1 / M2 / M3
[Hardware] GPU UUID / SM / L2 / register-SM / shared-SM / driver / CUDA toolkit / CUTLASS commit
[Permission] NCU hardware counters 是否实际可用；GPU 是否无其他人的计算任务
[Implementation] G0/G1/G2/G3/G4/G5 实现情况；CUTLASS 79a native MMA 证据
[Correctness] 量化 recipe、scale global/local、布局、参考计算、各误差指标
[Performance] core/full、warm/cold、BF16/NVFP4 主表、时钟/温度
[Resources] Tensor/FP32、Issue、SMEM、regs、spills、L2/DRAM、stall 的实测及原始报告
[Interference] 双侧减速、真实时间 overlap、makespan、probe progress、统计区间
[Hypothesis] 支持的差异 / 能解释的因果 / 反例 / 未确认
[Files] git commit + 原始 CSV/JSON + ncu/NSYS + SASS + 代码及复现命令
[Decision] PASS / CONDITIONAL / NO-GO（附依据）
```

**本文所有命令中 `./bin/gemm_bench`、`./bin/co_run_bench`、`./bin/micro_probe` 是要求开发者实现的程序接口，绝非服务器已有可执行文件；官方 79a 构建命令则可以直接作为 smoke test 起点。**
