# RTX 5090（SM120）BF16 vs 原生 NVFP4：资源瓶颈与并发干扰实验任务书

**版本：2026-11-14｜性质：实验实施任务书（待验证假设，不是已有实验结论）**

## 0. 研究目标和不可违反的规则

验证三个相互独立的命题：

- **Q1 资源画像：** 对同一逻辑 GEMM，BF16 和原生 NVFP4 的计算、访存、指令调度、寄存器/Shared Memory 与缓存行为如何变化？
- **Q2 因果诊断：** 两种精度对计算、L2、DRAM、Shared Memory、指令发射等受控压力的敏感性是否不同？这种差异能否归因于具体资源，而非不同 Kernel 实现、频率或缓存状态？
- **Q3 可利用性：** 在 BF16 和 FP4 主任务旁安排有用的辅助任务，是否能够取得真实的总完成时间收益，而不仅仅是让两个任务在时间线上重叠？

**反证同样算合格研究结果。** 不得预设 FP4 必然存在“空闲计算单元”；若没有稳定的精度相关差异，应如实报告。

必须区分以下三种对比：
1. **同一逻辑操作**：相同 M/N/K、相同原始激活/权重，使用 BF16 和由它们量化得到的 NVFP4。回答实际模型受益情况，但数据大小不可相同。
2. **硬件机制对照**：尽量在相似实现结构/相同算法阶段对照，同时用受控微基准剥离 Kernel 实现差异。不能把 cuBLASLt 和 CUTLASS 代码差异直接归因于精度。
3. **完整推理成本**：FP4 权重量化可以离线，但每轮需要的激活量化、Scale 处理、数据格式转换和额外 Kernel 启动必须计入；不能只报告 FP4 GEMM 核心时间作为端到端收益。

## 1. 平台及环境冻结（E0-A）

- GPU：NVIDIA GeForce RTX 5090，SM120，32GB；此前观测驱动 595.99.02，`nvidia-smi` 显示 CUDA 13.2 **只表示驱动支持上限，不保证已安装对应 CUDA Toolkit**。
- 不主动改变共享服务器的系统驱动、GPU 时钟、功耗上限、服务或其他用户会话。需要调整时钟先获得管理员许可。
- 先保存：`nvidia-smi -q`、`nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv`、`nvcc --version`、`ncu --version`、`nsys --version`、`gcc --version`、`cmake --version`、CUTLASS Git commit、`git status`。
- CUDA Toolkit 编译 SM120 的 CUTLASS 79a 案例要求 **CUDA >=12.8**。安装或使用一个可工作的 Toolkit（不要因为 `nvidia-smi` 显示 13.2 就以为 `nvcc` 已安装）。
- `ncu --list-sections` 和 `ncu --query-metrics` 保存本机实际可用指标；不可凭其他 GPU 的名字硬编码指标。验证有硬件计数器权限。
- 确认 GPU 没有其他计算用户正在运行；记录显示桌面进程 Xorg/gnome-shell 的显存占用，不要结束它们。
- 维持统一 GPU 时钟/热状态：预热至稳定；每批前后记录频率、功耗、温度、频率限制原因；BF16 与 FP4 交错运行，避免所有 BF16 跑完后才运行 FP4。

初始化建议命令：

```bash
mkdir -p ~/fp4_bf16_study/{src,bin,config,raw,profiles,figures,reports}
cd ~/fp4_bf16_study
nvidia-smi -q > raw/nvidia_smi_q.txt
nvcc --version > raw/nvcc_version.txt 2>&1 || true
ncu --version > raw/ncu_version.txt 2>&1 || true
nsys --version > raw/nsys_version.txt 2>&1 || true
ncu --list-sections > raw/ncu_sections.txt 2>&1 || true
ncu --query-metrics > raw/ncu_metrics.txt 2>&1 || true
```

## 2. 数据、精度与实现（E0-B）

**主基线为 BF16，不用 FP16 替换。** 数据准备：同一个随机种子生成或从真实 BF16 LLM 采集原始 `X[M,K]` 与 `W[K,N]`；使用这同一份 BF16 数据分别走 BF16 和原生 NVFP4 路径。先用固定随机正态/多尺度分布便于复现，再加入来自真实模型的一小批代表性激活和权重。记录生成器版本、随机种子、张量分布与布局。

- BF16：BF16 输入与权重、Tensor Core GEMM、FP32 累加、BF16 输出（最好统一 epilogue），保留强基线 cuBLASLt 和一套 CUTLASS BF16 实现。
- NVFP4：**W4A4**，激活和权重都为真 NVFP4 E2M1 存储格式，并附正确的 E4M3 每 16 个元素一个 Scale 及必要的全局 Scale；明确布局、Scale swizzle、对齐与算子支持。不能用 W4A16、FP4->BF16 解码后再 BF16 GEMM 冒充原生 FP4。
- FP4 基线：官方 `NVIDIA/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`。核实实际编译目标、使用本机 SM120 的 `mma.sync.aligned.block_scale` 路径及对应实际指令/计数器。示例输出 BF16，适合作为单 GEMM 路径。
- **准确性两层核对**：① 原始 BF16 张量在 FP32 精度下的 GEMM 参考值；② 将 NVFP4 数据按 Scale 正确反量化后产生的高精度 GEMM 参考值。BF16、FP4 输出分别对照对应参考，记录绝对误差、相对 L2、RMSE、cosine、P99 误差；另外记录 FP4 相对原始 BF16/FP32 参考的量化精度损失。不可因为 NVFP4 量化误差本来较大就跳过正确性。
- **核对异常**：Kernel 无输出、结果全零、NaN、错用 W4A16、输出读错、shape 填充错误，必须视为失败并修复。

### 固定工作负载

第一批：`A[M,4096] × B[4096,12288] -> C[M,12288]`，M 取 **16、32、64、128、256、512、1024**。

第二批（用于隔离尺寸和 L2 容量拐点）：固定 M=128/512，改变 K,N，选择至少 3 组权重工作集规模：两种精度均明显低于 L2、BF16 超过 L2 而 FP4 低于 L2、两种精度均超过 L2。通过 CUDA 设备属性读取 L2 大小，以实际 B 数据、Scale/布局的字节量判断；不能仅凭逻辑元素个数判断。记录是否经过 padding。

主系列的逻辑 FLOPs 均为 `2*M*N*K`，但真实执行的指令和字节要依据硬件计数器/Kernel 代码核对。

### Benchmark 统一接口（须自行实现，不是现成命令）

```bash
./bin/gemm_bench --dtype bf16|nvfp4 --m M --n N --k K \
  --input-source synthetic|model --seed 42 \
  --mode core|full --cache warm|cold --repeats 200 \
  --json raw/<name>.json
```

`core`：所有已打包数据在 GPU 内存，单纯 GEMM 执行；`full`：包含**每次都必须进行的**激活量化、Scale 准备、GEMM、必要格式转换，权重量化可预计算并明确标注。BF16 若存在额外转换也应计入。CLI 名称可以调整，但功能不可缺失。

## 3. E1：单 Kernel 资源画像

**目标**：测出不同 M 下的 BF16 与 NVFP4 的独立执行时间及资源特征，寻找瓶颈候选区域。**不要仅凭某个资源使用率推断闲置算力。**

对每个 M × dtype × cache 模式：

1. CUDA Event（建议配合 CUDA Graph）计时，预热 100 次、正式采样至少 100 次，报告 P10/P50/P90、均值/标准差；独立重复 3 次实验会话。小 Kernel 要使用批量 Graph 重放，扣除或摊销控制开销。
2. 用 Nsight Compute 分开抓 1–3 个代表 Kernel，优先使用本机存在的 `SpeedOfLight`、`LaunchStats`、`Occupancy`、`SchedulerStats`、`WarpStateStats`、`MemoryWorkloadAnalysis`、`ComputeWorkloadAnalysis` Sections，不可直接把 NCU 时间当成正常运行时间。
3. 记录：实际 Tensor/FP32 指令和 Pipe 活跃情况、DRAM 读写 Bytes/吞吐、L2 读写与命中、寄存器/线程、Shared Memory/CTA、CTA/SM、理论与实际 Occupancy、Eligible Warps、Issue Active、Warp Stall 原因、运行时钟、实际 Kernel 版本及 Tile/Stage 参数。
4. 缓存两组：`warm` 表示同一权重预热并重复使用；`cold` 使用大于 L2 的轮转/驱逐工作集或明确控制缓存，配合 NCU cache replay 设置验证。不要把每次调用新权重和原始量化进程混在一起。
5. FP4 同时报 `core` 和 `full` 时间，把激活量化与 Scale/格式准备成本独立拆出。

```bash
# 样例：这些 sections 仅在本机存在时使用，输出目录提前创建
ncu --section SpeedOfLight --section LaunchStats \
    --section Occupancy --section SchedulerStats \
    --section WarpStateStats --section MemoryWorkloadAnalysis \
    --kernel-name-base function --target-processes all \
    -o profiles/bf16_m128 \
    ./bin/gemm_bench --dtype bf16 --m 128 --n 12288 --k 4096 --mode core --cache warm --repeats 1
```

**工具陷阱**：NCU 默认 Kernel Replay 通常对 Kernel 串行执行并冲刷缓存，会改变时间/缓存状态。单 Kernel 热缓存实验在程序自行预热的前提下可用 `--cache-control none` 配合适用的 replay 模式；真实时间由不挂 NCU 的 CUDA Event 测得。对于并发实验主要用 Nsight Systems 和原生事件计时，不能用普通 NCU Kernel Replay 判断重叠。

**E1 输出**：
- `e1_baseline.csv`：至少字段 `run_id,dtype,M,N,K,mode,cache,kernel_id,tile,stages,workspace_bytes,weight_bytes,scale_bytes,time_us_p50,time_us_p90,achieved_tflops,regs_per_thread,smem_per_cta,occupancy,eligible_warps,issue_active,tensor_pipe,fp32_pipe,dram_read_bytes,dram_bw,l2_hit,l2_read_bytes,clock_sm,temp`，本机无指标则填 `NA` 说明，不准编造。
- 图 1：不同 M 的 BF16/FP4 `core` 和 `full` 时间，及 `core` 加速比；图 2：计算/访存/发射/Occupancy/寄存器等资源随 M 的变化。

## 4. E2：诊断瓶颈来源

### E2-A：计算资源敏感性

- 构造**独立计算微基准**：把 BF16/NVFP4 的 MMA 操作数加载到片上之后，改变可控 MMA 重复次数（1、2、4、8），保证运算不被编译器优化，记录真正执行的 MMA 指令数及时间；**这不是生产 GEMM 的等价变体**，只用于建立指令/流水线吞吐上限与计算压力响应。
- 对生产 GEMM 不擅自增加无用 MMA 后宣称其性能为“同一操作”，而是用独立基准建立计算带宽参考，再对照原 GEMM 的硬件指标。
- FP32 FMA 与 FP4/BF16 Tensor MMA 各自做压力微基准，记录占用的 Pipe、Instruction Issue、SMEM/寄存器；资源压力不纯净时需标注。

### E2-B：内存与缓存敏感性

- 同一 M/N/K：比较 `warm L2`、`cold/evicted` 和 `rotating weights` 3 个环境；计算运行时间增幅、L2 命中、DRAM 请求字节、DRAM 带宽。
- 做“尺寸拐点”对照：分别选择两种精度都可缓存、只有 FP4 可缓存、两者都不可缓存的权重集。这样可区分**压缩使权重放入 L2**与**计算单元使用变化**。
- 额外测试 BF16 和 FP4 的量化/Scale Swizzle 单独时间，不要把其成本隐藏在准备阶段。

### E2-C：内核配置与驻留敏感性

- 每精度先使用强基线的最优配置，再测试低寄存器/低 SMEM 或不同 Tile/Stage 的至少 2 个候选版本，记录独立时间、驻留 CTA 数和后续 E3 并发性能。
- 观察是否存在“降低资源占用但单 Kernel 变慢、并发总耗时反而降低”的情况；不能把 occupancy 提高直接当作加速证据。

**E2 输出**：`e2_sensitivity.csv`、计算压力曲线、热/冷缓存曲线、尺寸拐点曲线、最多 3 个明确且可证伪的瓶颈假设。每个假设列出支持和反对它的证据。

## 5. E3：双 Kernel 资源干扰矩阵（核心）

**主任务**：BF16 或 NVFP4 GEMM。首批重点 M=16、128、512；根据 E1 发现的分界可增加 M=64/256。主任务使用各自最佳 Kernel。

**辅助探针**（每个都要先单独 profiling，记录它实际占据的所有资源，不准声称纯粹只消耗一种资源）：

- P1：FP32 FMA（调节独立运算条数/ILP、寄存器数量）。
- P2：Tensor Core MMA（分别 BF16 和 NVFP4，调节吞吐压力）。
- P3：Shared Memory 读写（调节总访问次数、访问步幅和 Bank Conflict）。
- P4：L2 访问（使用可缓存工作集，调节访问密度和数据大小）。
- P5：GDDR7 streaming 读/写（大于 L2 的轮转数组，调节 CTA/带宽压力）。
- P6（可选）：高指令发射压力微基准，尽量低内存需求并分析实际 IPC。

每个探针分别校准**低/中/高**三个压力级别，记录探针在单独执行时的吞吐、活跃单元、使用寄存器/SMEM、执行时长。不要用“启动了多少个 Block”简单等同于“消耗了多少百分比资源”。

针对每个 `(M, dtype, probe, pressure)` 依次测量：

1. `A-alone`：只有主 GEMM，记录耗时 `T_A`。
2. `B-alone`：只有探针，记录耗时 `T_B`。
3. `serial A→B`：两个任务按顺序运行，记录总时间 `T_serial`。
4. `concurrent A||B`：同进程双 CUDA Stream，在同一可复现触发条件后启动，记录 `T_A_conc`、`T_B_conc`、`T_joint`。
5. 记录 Nsight Systems 时间线，**确认确有时间交叠**；如果没有重叠，不能把性能变化归因于运行时硬件资源竞争。

短 GEMM 容易因为启动时间而难以交叠。可把两个独立任务各自封装在重复的 CUDA Graph 节点序列中，使总测量窗口约 10–50 ms；明确记录 Graph 的执行时间与真实的节点重叠比例。**不能凭重复 launch 假定同 SM 驻留**：不同 Kernel 可以在不同 SM 上运行；同 SM 竞争要结合 CTA 资源模型和可用硬件证据推断。是否可以控制分区，须先确认该设备/环境 API 支持。

计算至少四个指标：

```
Main slowdown = T_A_conc / T_A
Probe slowdown = T_B_conc / T_B
Overlap gain = T_serial / T_joint
Relative sensitivity = Main slowdown_FP4 / Main slowdown_BF16
```

注意 `Relative sensitivity` 只是相对减速敏感性；由于 FP4 与 BF16 的主任务本身速度不同，**还必须报告绝对时间节省、每秒主任务完成次数和辅助任务完成次数**。两种精度的探针工作量需相同，并额外测试控制探针时长相对主任务的比例，否则“FP4 干扰更小”可能仅因 FP4 主任务过早结束。

建议先跑**12 个探索性条件**：3 个 M（16/128/512）× 2 种精度 × 2 个探针（FP32 和大规模 DRAM Streaming），每个探针 3 档压力＝36 个组合；再扩充共享内存、L2 和 Tensor Core 探针。3 个独立会话交错重复。这样避免一开始就进行数百甚至上千个无效组合。

**E3 输出**：
- `e3_interference.csv` 包含 dtype/M/probe/pressure、独立延迟、并发各自延迟、联合时间、减速比、加速比、重叠比例、功率/频率与 95% 置信区间。
- 图 3：`dtype × probe` 的主任务减速热力图（分 M 单独绘制）。
- 图 4：横轴辅助任务进度/吞吐，纵轴主任务减速，BF16 与 FP4 各一条帕累托曲线。
- 图 5：最显著案例的 Nsight Systems 时间线截图/原始报告。

## 6. E4：证明“资源差异具有实际价值”

不要一开始就写复杂新调度器。先做两个不改变依赖的应用级验证：

**E4-A：同一层 FFN 中独立分支**

Gated FFN：

`G = X W_gate; U = X W_up; Z = SiLU(G) * U; Y = Z W_down`

`G` 和 `U` 在输入 X 已经就绪后可以并行；`SiLU(G)*U` 必须等二者完成；`W_down` 必须等 Z 就绪。

逐精度测试：
- 串行 `gate→up→activation→down`；
- 两个 Stream 并发 `gate||up→activation→down`；
- **强基线**：如果有可用的融合 gate+up / fused FFN Kernel，必须加入。不能只战胜人为拆分的低效串行方案。
- FP4 的两次激活量化能否复用，必须控制；完整路径包括所有在线 Scale/格式准备成本。

记录完整 FFN 延迟和 BF16、FP4 独立路径优化收益。

**E4-B：有用的准备工作**

只选与当前 GEMM 没有数据依赖的真实工作：例如下一个请求中已经准备就绪的数据布局转换、另一个请求的元数据准备、下一层权重的预触达/预取（需验证是否导致 L2 污染）。**不能声称尚未计算出的下一层激活可以被提前量化。**

同精度下比较 `serial`、`concurrent`、合适的融合/调度强基线，报告主任务耗时、准备工作完成量及总 makespan；检查是否因 L2 污染使后续层变慢。

**E4 输出**：`e4_ffn.csv`、`e4_prepare.csv`，以及明确回答“FP4 相比已经优化过的 FP4 基线是否获得额外收益”。

## 7. 统计、可重复性与质量门槛

- 每个条件随机交错 BF16/FP4 测试顺序；至少 3 个独立会话，每会话不少于 100 个计时样本，Graph 累积计时需报告每调用均值及样本结构。统计 P50/P90、均值、标准差、95% bootstrap CI；不以单次最优值作比较。
- 观测到 >5% 的变化才能优先列入复核候选，但 5% 不是统计显著性的替代。对所有关键发现增加独立重复、检查置信区间是否跨 1、检查时钟/功率/温度变化。
- 单独最快 Kernel 的参数、并发最优 Kernel 的参数均完整记录。必须为 BF16 和 FP4 分别调优，不应故意让某一精度使用差实现。
- 在硬件计数器、受控压力、执行时间三种证据不能相互印证时，写“未确认瓶颈”，不允许硬判定为计算/带宽受限。
- 如果 NCU 内核计数器缺失，记录本机为何不可用；不能用 GPU-Util、Occupancy 代替 Tensor Core 利用率。
- 不得在共享服务器上停止桌面进程、改驱动、强行抢占别人的 GPU、未经许可锁定全局功率和时钟。

## 8. 必须交付的工程文件

```
fp4_bf16_study/
├── README.md               # 一键复现方法；系统依赖和环境锁定
├── config/                 # 固定精度、形状、seed、cache、kernel 参数
├── src/
│   ├── gemm_bench.*        # BF16/NVFP4 core/full 与正确性验证
│   ├── quantize_nvfp4.*    # 原生格式/scale 数据准备
│   ├── probe_*.cu          # FP32 / Tensor / Shared / L2 / DRAM 探针
│   └── interference.*      # 双 Stream 启动与计时
├── scripts/
│   ├── env_check.sh
│   ├── run_e1.sh
│   ├── run_e2.sh
│   ├── run_e3.sh
│   └── run_e4.sh
├── raw/                    # CSV/JSON、日志、SASS/PTX 检验、环境记录
├── profiles/               # .ncu-rep 与 .nsys-rep 原始报告
├── figures/                # 清晰标注单位的图表
└── reports/
    ├── findings.md         # 各个假设：证据/反证/不确定性
    └── reproducibility.md  # 完整复现过程与限制
```

所有原始 CSV/JSON 不得只留图片；每张图的绘制脚本必须附上。所有实验失败、unsupported shape 或无法采集的计数器须留日志。输出必须能在相同设备上复现，不能只提供口头结论。

## 9. 执行顺序与阶段性验收

### 里程碑 1：E0 + E1（先做这一步）

成功条件：
1. BF16、原生 NVFP4 都有可运行、数值可核查的实现，并保存其源码及 commit。
2. 能证明使用 **SM120 原生 NVFP4 MMA**，而非反量化到 BF16 的模拟实现。
3. 每个 M、热/冷缓存下有正常事件计时和核心 NCU 指标；FP4 分开报告 core/full。
4. 给出两个最值得继续的瓶颈假设，以及反例/可能的实现差异。

### 里程碑 2：E2 + E3

成功条件：
1. 至少 1 个缓存敏感性、1 个计算敏感性受控实验。
2. 至少 2 种资源探针 × 3 个 M × 2 精度的完整干扰记录，存在确认过的真实时间交叠。
3. 有主任务和辅助任务的双侧减速、总 makespan、硬件指标以及跨会话置信区间。
4. 明确区分“精度导致的计算/访存变化”与“采用不同 Kernel 实现导致的变化”。

### 里程碑 3：E4

成功条件：
1. 至少完整 FFN 或一类真正有用的无依赖准备任务。
2. 对比最强可用串行与融合基线，计算完整路径收益。
3. 如没有额外收益，明确提交负结果以及失败原因。

## 10. 供执行者回复的固定格式

每个里程碑完成后，回复以下内容：

```
环境：驱动/CUDA Toolkit/CUTLASS commit/NCU/NSYS/频率策略
精度路径：BF16 Kernel 名称；NVFP4 Kernel 名称；native MMA 证据
准确性：误差指标与对应参考
核心数据：BF16、NVFP4 core/full P50/P90（按 M）
资源证据：Tensor、FP32、Issue、DRAM、L2、Registers、SMEM、Warp Stall
干扰证据：A 单独/B 单独/A||B 的时间和进度、时间线及统计区间
支持的假设：列 1-3 个，指出直接证据
反对的证据：列反例和不可解释项
局限性：缺失指标、不同 Kernel、缓存、频率及共享 GPU 干扰
复现命令：可直接复制执行；附件 CSV、报告和代码位置
下一阶段选择：继续/修正假设/停止，并给出依据
```

## 参考实现与工具文档

- 官方 SM120 NVFP4/BF16 CUTLASS 79a 示例：<https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu>
- NVIDIA Transformer Engine NVFP4 格式：<https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html>
- NVIDIA Nsight Compute Profiling Guide（Kernel Replay、Range Replay、Clock/Cache 控制）：<https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- NVIDIA Nsight Compute CLI：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/>
- NVIDIA Nsight Systems User Guide：<https://docs.nvidia.com/nsight-systems/UserGuide/>
