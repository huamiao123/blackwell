from pathlib import Path
import json,statistics,datetime
ROOT=Path(__file__).resolve().parent/'delivery'
b=json.loads((ROOT/'report_data.json').read_text())
S=[s for s in b['summaries'] if s.get('protocol')=='default_scheduler_v2']
B=[d for d in b['raw_briefs'] if d.get('protocol')=='default_scheduler_v2' and '_baseline_' in d['file']]
def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def cfg(d):return d['args']['batch'],d['args']['precision'],d['args']['mode']
def base(batch,prec,mode):
    return statistics.median([r['mean_step_ms'] for d in B if cfg(d)==(batch,prec,mode) for r in d['runs']])
def metric(s,name):return next((m for m in s['metrics'] if m['name']==name),{})
baseline_rows=[]
for d in sorted(B,key=lambda d:(-d['args']['batch'],d['args']['precision'],d['args']['mode'])):
    runs=d['runs'];batch,prec,mode=cfg(d)
    baseline_rows.append([batch,prec,mode,len(runs),fmt(statistics.median(r['decode_wall_ms'] for r in runs)),
        fmt(statistics.median(r['mean_step_ms'] for r in runs)),fmt(statistics.median(r['p50_step_ms'] for r in runs)),
        fmt(statistics.median(r['p95_step_ms'] for r in runs)),fmt(statistics.median(r['output_tokens_per_s'] for r in runs),1)])
speed_rows=[]
portrait_rows=[]
for batch in [8,1]:
    if not any(d['args']['batch']==batch for d in B):continue
    for prec in ['bf16','nvfp4']:
        e,g=base(batch,prec,'eager'),base(batch,prec,'graph')
        speed_rows.append([batch,prec,fmt(e),fmt(g),fmt(e/g)+'×',fmt((1-g/e)*100,2)+'%'])
        runs=[s for s in S if s['args']['batch']==batch and s['args']['precision']==prec and '_graph_metrics_' in s['trace']]
        values=[]
        for name in ['DRAM Read Bandwidth [Throughput %]','SMs Active [Throughput %]','Tensor Active [Throughput %]']:
            v=[metric(s,name).get('mean') for s in runs];v=[x for x in v if x is not None]
            values.append(f'{min(v):.2f}–{max(v):.2f}' if v else '—')
        portrait_rows.append([batch,prec,fmt(g),*values])
metric_rows=[];coverage_rows=[];sampling_rows=[];native_rows=[];gap_rows=[]
for s in sorted(S,key=lambda s:(-s['args']['batch'],s['trace'])):
    if '_pilot_' in s['trace']:continue
    tag=f"B{s['args']['batch']} {s['args']['precision']} {s['args']['mode']}"
    if '_metrics_' in s['trace']:
        for name in ['SMs Active [Throughput %]','SM Issue [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]','DRAM Write Bandwidth [Throughput %]','Compute Warps in Flight [Throughput %]']:
            m=metric(s,name)
            metric_rows.append([tag,s['trace'].rsplit('_',1)[-1],name.split(' [')[0],fmt(m.get('mean')),fmt(m.get('p50')),fmt(m.get('p95')),fmt(m.get('peak')),m.get('valid_samples','—'),fmt(m.get('active_mean')),fmt(m.get('gap_mean'))])
        sm=s['sampling'];sampling_rows.append([tag,s['trace'].rsplit('_',1)[-1],sm['count'],fmt(sm['dt_p50_ns']/1000,3),fmt(sm['dt_p95_ns']/1000,3),fmt(sm['dt_max_ns']/1000,3),sm['intervals_gt_1_5x_median']])
    if s['kernel_coverage_reliable'] and ('_metrics_' in s['trace'] or '_node_' in s['trace']):
        coverage_rows.append([tag,'node/32' if '_node_' in s['trace'] else 'metrics/224',s['analyzed_steps'],fmt(s['window_ms']),fmt(s['kernel_union_ms']),fmt(s['gap_ms']),fmt(s['kernel_coverage']*100,2)+'%',s['kernel_count']])
        gap_rows.append([tag,fmt(s['gap_ms']),fmt(s['gap_overlapping_launch_api_ms']),fmt(s['gap_overlapping_sync_api_ms']),fmt(s['gap_p50_us']),fmt(s['gap_p95_us'])])
    if s['native_sm120_e2m1_kernel_instances']:
        native_rows.append([tag,s['trace'].rsplit('_',1)[-1],s['native_sm120_e2m1_kernel_instances'],s['analyzed_steps'],s['engine_core_pid_mapping_match']])
overhead_rows=[[r['batch'],r['precision'],r['mode'],r['trace'].split('_')[4],r['steps'],fmt(r['profile_step_ms']),fmt(r['baseline_median_step_ms']),fmt(r['vs_baseline_pct'],2)+'%' if r['vs_baseline_pct'] is not None else '不同长度，不作比值'] for r in b['overhead'] if '_pilot_' not in r['trace']]
report=f'''# RTX 5090 原生 NVFP4 vs BF16：NSYS GPU Metrics 第二轮报告

整理时间：{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}。服务器产物目录：`~/fp4-llm/results/nsys_round2/`。

## 结论与证据等级

本轮把实际请求的无 Profiling 性能、NSYS 时间线和设备级 GPU Metrics 分开测量。当前配置已经观察到 BF16/NVFP4 的资源使用差异，但**没有证明低利用率可以转化为安全、有效的辅助计算容量**。

| 假设 | 判断 | 证据与限制 |
| --- | --- | --- |
| H1：BF16 Decode 的 DRAM 使用高于 NVFP4 | **已验证，限定本次 Graph 采样条件；Eager 为初步支持** | Graph 有独立进程复测、真实设备映射及稳态窗口；比较的是设备级时间平均，不是单 Kernel 的专属吞吐，也不是完整模型每步 DRAM 字节。 |
| H2：FP4 Kernel 缩短后，Eager 间隔比例增大 | **初步支持** | 同样 NSYS 条件下，FP4 的观测 Kernel 并集占比更低；但 Eager 的 CUDA 时间线采集本身显著增加延迟，不能把该间隔比例直接当正常运行比例。 |
| H3：Graph 减少执行间隔并改善 FP4 性能 | **性能部分已验证；间隔的因果解释为初步支持** | 每配置五次真实请求测量及实际 Replay 记录支持性能改善；Graph 路径同时涉及正常编译/融合、异步调度和 Replay，不能把全部差额归于 Graph Launch 单一机制。 |

下面的“已验证”只对应本机、指定模型与测试条件。基线五次请求在同一个已加载模型进程中执行；不是五次独立模型初始化。Graph Metrics 的 r02/r03 是独立进程采集。Eager 正式 Metrics 只保留一轮，不用反复采集来制造高置信度。

### 关键数据速览

下表性能列来自无 Profiling 基线；资源列来自两次独立 Graph＋10 kHz 采集的稳态均值范围。两类测量没有混算成加速比。

{table(['Batch','精度','正常Graph ms/步','DRAM Read均值 %','SMs Active均值 %','Tensor Active均值 %'],portrait_rows)}

一个具体的执行路径差异是：32步Node记录中，BF16 Batch=1的Linear为cuBLAS `internal::gemvx::kernel`族，共4640次（含lm_head）；Batch=8则记录到BF16 WMMA TensorOp族，共4640次。NVFP4在两个Batch中都记录到4608次SM120 E2M1 GEMM，对应每步144个量化Linear，另保留未量化lm_head。**不能假定BF16和FP4在所有小M上使用同类Tensor Core数据流。** Batch=1时BF16 Tensor Active很低、FP4反而更高，与上述执行族差异相符；这里没有凭Kernel名称推断未检查的完整SASS或所有CUDA Core指令。

## 1. 完成范围与执行记录

- 8 个核心配置：Batch 1/8 × BF16/NVFP4 × Eager/Graph，每配置 5 次无 Profiling 基线。
- Batch=8 先完成小规模采集、设备/时间验证，再进行正式基线与采集；随后扩展 Batch=1。
- 各核心配置分别采集仅 CUDA/NVTX 时间线和时间线＋10 kHz Metrics；Graph 另做独立 Metrics 复测、32 步 Node 级短窗口。
- 没有运行 NCU、辅助 Kernel、Kernel 优化、时钟调整或功率调整。
- 每个 GPU 作业启动前检查 `nvidia-smi -q`、`gpu-who`、计算进程与已知自动启动控制器；运行时每约 2 秒检查计算进程，有硬超时。未抢占或终止其他用户进程。
- 全命令和显式环境变量见 `commands.txt`；逐次 pre/post-check、运行输出及监督状态见 `logs/`。
- 未覆盖历史实验报告。原始 `.nsys-rep`、SQLite、逐步输出 JSON 均保留。

**诊断控制组不混入正式基线。** 最初为逐步计时试用了 `async_scheduling=False`；它明显改变延迟，因此 r01 的四个同步调度基线只保留在 `analysis/baseline_sync_diagnostic.csv`。正式脚本恢复历史默认异步调度，原脚本归档为 `scripts/decode_round2_sync_diagnostic.py`，正式原始 JSON 标记 `protocol=default_scheduler_v2`。

一次 Graph pilot 在 NSYS 结束时，目标 Python 短暂被重新托管，监督器曾把它误判为“新计算任务”。原始 PID 与该 pilot JSON/Trace 中的自身 PID 一致；没有其他用户任务。已在 CPU 侧定位并改为保留已登记的后代 PID；保留原日志，pilot 不作为正式性能结果。未因该误报重新运行模型。

## 2. 环境与固定工作负载

| 项目 | 设置 |
| --- | --- |
| GPU | RTX 5090 / GB202 / SM120，170 SM，32 GB；UUID `022c5f55-490e-ec54-3887-5f12510d273d`，PCI `0000:27:00.0` |
| 软件 | vLLM 0.33.0、torch 2.14.0+cu132、FlashInfer 0.7.2.post1、NSYS 2026.1.3.425；系统工具链 CUDA 13.3，驱动报告 CUDA 13.2 |
| 模型 | 本地 Qwen3-8B-BF16 / Qwen3-8B-NVFP4；沿用已校验的模型 |
| 提示词 | 固定 2048 个 Token ID：`1000 + i % 5000`，全文见 `input_token_ids.json` |
| Batch | 1 或 8；8 个真实活跃请求，逐步调度记录检查实际 Batch，未用固定形状 bench forward 替代 |
| KV | 两种精度均为 `fp8_e4m3`；上下文随 AR 解码自然增长，并非每步冻结为 2048 |
| Attention | 显式固定 FLASHINFER；运行日志记录 XQA Decode、BF16 Query 与 FP8 KV |
| Linear | 两边统一 `linear_backend=cutlass`；NVFP4 的运行时 Kernel 记录包含 SM120 block-scaled E2M1 模板 |
| 调度 | 同进程 `InprocClient`，`VLLM_ENABLE_V1_MULTIPROCESSING=0`；正式数据使用默认异步调度 |
| 其他 | max_model_len=4096、max_num_batched_tokens=8192、max_num_seqs=8、gpu_memory_utilization=0.85、Graph capture sizes=[1,8]、seed=0 |
| 采样 | temperature=0、ignore_eos=True；每请求 257 个输出 token：首个由 prefill 产生，随后 256 个实际 Decode Step |

这是**离线 `LLM.generate` 真实请求的 CPU＋GPU Decode 计时**，包含当前进程的 EngineCore 调度与输出处理；不包含 HTTP、网络、完整服务排队或首个 prefill。不是在线服务吞吐基准。

两边固定提示 Token ID；生成 token 由各自模型预测，未强制为同一轨迹。原始 JSON 保存输出长度、生成序列 SHA256、逐步实际 Batch 与 scheduled tokens。Batch=8 的请求使用同样提示，启用了 prefix caching，因此其 KV 具有共享前缀局部性；不能推广为八条完全无关请求。

每进程先预热两次 33-token 生成。正式窗口在第一次真实 Decode 被调度、前置 GPU 工作完成后开始；只在窗口边界 synchronize，未逐步额外同步。Step P50/P95 使用 CPU 接收到完整批次输出进度的时间间隔；异步流水下它不是单个 GPU Kernel 的服务时延。脚本保留少量相同的 CPU/NVTX 观测 hook，其开销未单独消融。

## 3. 无 Profiling 正式基线

无 NSYS、NCU 或 torch.profiler。下表各项为五次请求测量的中位数；“256步墙钟”排除首个 prefill，Tokens/s 计算 `Batch×256 / Decode墙钟`。每次请求实际执行到 257 个输出 token，调度记录确认其中 256 次为单 token Decode。

{table(['Batch','精度','模式','请求重复','256步墙钟 ms','平均步 ms','P50步 ms','P95步 ms','输出 tokens/s'],baseline_rows)}

完整逐次数据见 `baseline.csv`，逐步时延见 `analysis/decode_step_stats.csv`。显存、SM clock、功耗和温度保存在 `analysis/hardware_telemetry.csv`，原始 `nvidia-smi -q` 在监督日志中。遥测约每两秒一次，包含初始化/预热，不能冒充每个 Decode Step 的硬件状态。

### Graph 相对 Eager 的实际性能

{table(['Batch','精度','Eager ms/步','Graph ms/步','加速比 E/G','延迟降低'],speed_rows)}

Eager Graph Launch 数为 0；正式 Graph Capture 日志、Python Replay 计数与 SQLite `cudaGraphLaunch` 均检查过，每次正式窗口记录 256 次 Replay，实际 Batch 与指定值一致。没有把 `enforce_eager=False` 本身当作实际启用的证据。

## 4. Profiling 扰动：必须先看此表

{table(['Batch','精度','模式','采集','步数','采集 ms/步','基线 ms/步','相对基线'],overhead_rows)}

Eager 的主要扰动来自 CUDA 时间线跟踪本身；10 kHz Metrics 的增量应与同配置 timeline 比较。Graph 级采集影响较小。发现 Eager 扰动后，相关数据降为采样条件下的诊断，不用于估计正常推理的精确间隔；没有提高到 20 kHz 或反复重跑该路径来强行确认正常时序。其余采集用于完成有/无 Metrics 的扰动定位及低开销 Graph 对照。

Node 级只有 32 步，未与 256 步基线计算精确扰动比；它用于执行组成与观测 Kernel 覆盖。**不将 Node Profiling 耗时与未 Profiling 的 Eager 耗时直接计算加速。**

## 5. 时间戳与设备身份核查

1. 先通过实际 Schema 确认：`GPU_METRICS.timestamp`、CUPTI `start/end`、NVTX `start/end` 使用 Trace 的 ns 时间；全程没有使用 rawTimestamp 过滤 Kernel，也没有自行添加偏移。
2. `GENERIC_EVENT_SOURCES.timeSourceId` 对应本机枚举的 `SessionNs`；GPU Metrics、Kernel/Graph 和 Decode NVTX 的范围相交。窗口前 SM/Tensor 活动低、进入 Decode 后上升，与时间线对应。
3. 采集入口的设备 0 已由 NSYS help 映射到指定 PCI 地址；再通过 GPU inventory UUID 与 `TARGET_INFO_CUDA_DEVICE` 的 UUID/进程关系确认是同一设备。没有只凭两个索引都为 0 判断身份。
4. EngineCore 使用 InprocClient，GPU Kernel 所属 CUDA 进程与脚本 PID 对应；标记没有停留在一个不执行 CUDA 的父进程。
5. 历史数据实测采样间隔约 1 ms，历史启动脚本也显式设置了 1000 Hz；因此旧 1 kHz 是实际采集设置。历史“过滤后只剩约31%采样点”本身不证明时钟错位，与约29%观测覆盖可以相容。

详见 `analysis/timestamp_validation.json` 和逐 Trace validation JSON，其中保留 Schema、范围、UUID、采样统计及 Diagnostics。

### 10 kHz 实际采样质量

{table(['配置','重复','整段样本数','间隔P50 μs','间隔P95 μs','最大间隔 μs','>1.5×中位间隔数量'],sampling_rows)}

未观察到指标采样间隔缺口或 Buffer Overflow/Missing Data 诊断。NSYS 保留了 capture 边界相关的 “Not all NVTX/CUDA events might have been collected” 警告；不能把“无 overflow”写成工具绝对完整性证明。目标 Decode 标记、Replay 数及主要 Kernel 计数的对应关系用于检查目标窗口。

GPC/SYS Frequency 在 SQLite 中出现负数或远超合理 MHz 范围的值，均标为无效，未取绝对值、未猜测换算比例。该频率字段不参与推理；硬件 clock 另保留 `nvidia-smi` 遥测。

正式采集为 256 步；稳态指标和主要覆盖统计使用输出完成标记之间的 **第17–240步，共224步**，剔除开始/结束各16步。32步 Node 诊断不作该裁剪。性能扰动对照仍使用完整256步原始墙钟，不能混用两个分母。

## 6. 设备级指标：稳态窗口统计

所有百分比均沿用 NSYS 指标名的 `% of peak`，不换算成 FLOPs。下表同时提供全窗口均值/P50/P95/峰值；峰值只作参考。Graph 级 Trace 无完整 Node 活动时，活跃/间隔均值留空。Eager/Node 的活跃分类按采样点 timestamp 与观测 Kernel 并集相交，100 μs 采样不能变成单个短 Kernel 的专属计数器。

{table(['配置','重复','指标','全窗均值','P50','P95','峰值','有效样本','活跃点均值','间隔点均值'],metric_rows)}

活跃/间隔的各类有效采样数、无效采样数保存在 `analysis/metric_summary.csv`。原始设备样本见 `analysis/metric_samples.csv`，所有指标还保留在逐 Trace metrics CSV。

## 7. Kernel 时间并集与间隔

逐设备、逐目标窗口裁剪区间后排序合并。`T_active` 是观测 Kernel 区间并集；`T_gap=T_window−T_active`。该比例称为 **Kernel time coverage**，不称为 SM 利用率，也不把间隔全归因于 CPU。

{table(['配置','采集/窗口','分析步数','窗口 ms','Kernel并集 ms','无观测Kernel ms','覆盖率','Kernel记录数'],coverage_rows)}

Graph 级 Trace 中的 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` 仅为 Graph 外包络，内部可能有空隙，因此该类正式256步采集的 Kernel Coverage 标为不可可靠计算。图4中的 Graph 分解来自单独32步 Node Trace，不能回填到低开销 Graph 级运行。

### 与 CPU Launch/同步的关系

{table(['配置','总间隔 ms','与Launch API重叠 ms','与同步API重叠 ms','单间隔P50 μs','P95 μs'],gap_rows)}

这些是**时间重叠关联**，不是已证明的因果份额；Launch 与同步可能重叠，不能把两列直接相加。CPU_SCHEDULER NVTX 与 CUDA Runtime correlationId 提供调用证据，但没有 OS context-switch 采样或完整依赖停顿证据，剩余部分保持“未归因”。不能由 `window−kernel_union` 得出“CPU造成了全部等待”。

### 运行时原生 NVFP4 路径证据

{table(['配置','重复','观测SM120 E2M1 GEMM次数','分析步数','进程映射一致'],native_rows)}

完整 Kernel 名保存在逐 Trace summary JSON，包含 `SM120_16x8x64_TN_VS`、`float_e2m1_t` 与 block-scaled CUTLASS 类型；该证据来自实际运行，而非只检查二进制里存在某个符号。本轮未使用 NCU，也没有据此声称取得动态 MMA 指令计数。

## 8. 四张核心图

![图1：GPU时间线](figures/01_gpu_timeline.png)

图1分开画 Graph 外包络和观测 Kernel；图中空白仅描述对应 Trace，不能泛化成可插入任务的空间。

![图2：DRAM Read时间序列](figures/02_dram_read_timeseries.png)

图2展示1 ms显示分箱均值；原始10 kHz数据保留。横轴是实际 Trace 时间，各次运行长度不同。

![图3：SMs Active和Tensor Active](figures/03_sm_tensor_timeseries.png)

![图4：Decode时间分解与正常基线](figures/04_decode_time_breakdown.png)

图4左侧在各自 Profiling 窗口内分解，右侧单独给出无 Profiling 基线；不混算两类时间。

## 9. 对任务书八个问题的回答

1. **稳定 Decode 时间是多少？【已验证，限本配置五次请求重复】** 见第3节，每种模式保留平均步时延、P50/P95及所有原始重复；不使用最优一次作为正式结果。
2. **Graph 对两种精度提升多少？【性能部分已验证】** 见第3节，FP4 的默认 Graph 路径改善明显更大；单独 Graph、编译融合与调度各自的因果贡献未消融。
3. **BF16 DRAM 是否长期接近峰值？【初步支持较高平均使用；无法确认持续饱和】** Graph 稳态均值/P95比单个87%峰值更有意义，但不足以证明唯一瓶颈。Eager 空隙会稀释设备均值。
4. **FP4 DRAM 使用是否降低？【已验证，限Graph设备级统计】** 本次两个Batch的重复采集直接比较；不将此等同于每个Kernel服务速率降低，或宣称剩余带宽可以无干扰复用。
5. **Tensor/SM 有何差异？【已验证，限采样条件】** 见第6节；FP4可出现更低SM活跃而更高Tensor Active。不同格式下指标不是统一的剩余FLOPs比例。
6. **多少间隔可归因CPU？【无法给出确定份额】** 第7节只给Launch/同步时间重叠及未归因部分。Eager tracing改变时间分布，不能声称其余百分比全在等CPU。
7. **Profiling是否改变现象？【已验证存在扰动】** Eager CUDA tracing的延迟增加明显；Metrics增量和Graph级采集分别对照。Eager数据只作受扰条件下的现象，不用来定量描述正常间隔。
8. **是否证明FP4有可利用硬件余量？【无法判断】** 没有辅助任务准入、共驻留、干扰或兑现收益实验；此轮按任务书不做这些实验。只能确认资源画像与执行组织不同。

## 10. 未完成、未验证及适用边界

- 没有20 kHz采集：10 kHz已满足设备级稳态统计，不为追求更高频率增加扰动；仍不能解析每个数十微秒Kernel的独占资源使用。
- 没有为Eager构造零扰动的完整Kernel时间线；其正常运行Coverage和严格CPU因果份额仍无法判断。该项不能靠再跑同样的Trace解决。
- 没有完整在线服务请求混合、跨模型、跨GPU或模型准确率评估。
- FP8 KV加载日志仍有scale dtype/默认1.0警告。本轮保持既有配置并记录，不据此宣称质量等价，也未改动checkpoint或scale。
- 两个模型同时改变权重/激活数值与执行实现；这不是只更改权重位宽的单变量实验。
- 实际Kernel资源准入（寄存器、SMEM、CTA驻留）和另一个任务能否利用余量，留给后续决策。本轮数据不自动授权下一轮并发实验。

## 11. 文件索引与复核入口

`environment.txt` / `environment_check_report.md`：NSYS本机帮助、支持集合、历史Schema核查；`commands.txt`：完整启动命令与环境；`baseline.csv`：正式性能基线；`raw/`：JSON及nsys-rep；`sqlite/`：原始导出；`analysis/`：验证、设备样本、区间、步时延、指标统计及扰动表；`figures/`：PNG和SVG。

实验脚本位于 `~/fp4-llm/scripts/decode_round2.py`、`run_one.py`、`run_plan.py`、`validate_trace.py`、`analyze_round2.py`、`collect_bundle.py`。脚本SHA256和数据清点保存在 `analysis/report_data.json` 与 `analysis/artifact_manifest.json`。
'''
(ROOT/'final_report.md').write_text(report,encoding='utf-8')
environment='''# NSYS第二轮环境与既有资产检查

本机NSYS版本2026.1.3.425。授权包装器可列出设备0：RTX5090 GB202、PCI0000:27:00.0；可用集合为gb20x及gb20x-top，正式选择gb20x。参数帮助确认graph/node granularity、cudaProfilerApi捕获、stop-shutdown、10kHz采样受支持。

历史prof/gpumetrics_{bf16,nvfp4}.sqlite内确有GPU_METRICS、TARGET_INFO_GPU_METRICS、CUPTI_KERNEL、RUNTIME、NVTX、GPU及CUDA_DEVICE映射；字段已用PRAGMA核对。历史实测样本间隔中位数约1ms，历史脚本显式设置1000Hz。旧DECODE_ONLY标记只覆盖33-token生成且包含首步，因此重新限定本轮Decode窗口。

复用原decode_prof2.py/decode_graph.py的LLM启动路径，新增的是请求计数、实际调度观察和标记，不是模型或Kernel优化。正式EngineCore为同进程InprocClient；Graph有实际Replay及cudaGraphLaunch记录，Eager没有。NVFP4实际Kernel名确认进入SM120 E2M1 block-scaled CUTLASS路径。两模型Attention均固定FlashInfer XQA、FP8 KV。

所有命令和实际输出存environment.txt及logs/environment_*.txt；historical_schema.json保存旧库Schema。正式逐库Schema和身份/时间范围保存在analysis/*_validation.json。授权包装器接受所需NSYS参数；TMPDIR单独preserve可用，而同时preserve TMP/TEMP被sudo拒绝，该探测没有启动模型。应用缓存和脚本输出定位在项目内；NSYS日志仍显示工具内部/tmp会话暂存，未声称该内部路径已成功重定向，也未在项目外手动清理。

未调用NCU。未变更驱动、权限、GPU频率或功率限制。当前条件下Eager tracing有明显扰动，其后续解释见final_report.md，不视作正常性能画像。
'''
(ROOT/'environment_check_report.md').write_text(environment,encoding='utf-8')
print('REPORT_WRITTEN',len(report),'characters')
