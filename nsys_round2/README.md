# Qwen3-8B Decode：NSYS GPU Metrics 第二轮（2026-11-16）

本目录记录真实模型在 RTX 5090 / SM120 上的 BF16 与原生 NVFP4 对照。
它与仓库中的合成 GEMM 干扰实验独立：没有运行 NCU、新的辅助 Kernel 或 Kernel 优化。

先看 [完整报告](final_report.md)，再看 [正式基线](baseline.csv) 和
[Profiling 扰动表](analysis/profiler_overhead.csv)。

## 已完成的测量

- Batch=1/8、提示长度2048、256个真实Decode Step，BF16/NVFP4 × Eager/Graph，共8个配置。
- 每配置5次无Profiling请求测量，共40条正式基线；单进程内的请求重复不冒充独立模型初始化。
- 每配置各一份仅CUDA/NVTX时间线与一份10 kHz Metrics采集。
- Graph Metrics另有独立进程复测；Graph Node另采32步短窗口。
- 含小规模pilot在内，共26份Trace/SQLite；启动命令、输入、原始JSON和诊断日志保留。

| Batch | BF16正常Graph ms/步 | NVFP4正常Graph ms/步 | Graph采样DRAM Read均值：BF16 / NVFP4 |
| --- | ---: | ---: | --- |
| 1 | 10.024 | 6.725 | 约86.4% / 44.2%–44.4% |
| 8 | 10.962 | 6.880 | 约79.3% / 44.0% |

性能列来自无Profiling基线；资源列来自独立采样，不能混算成加速比。
Eager CUDA tracing本身有明显扰动，其Kernel间隔只作采样条件下的诊断。
**当前证据确认资源使用差异，尚未证明存在可无干扰利用的硬件余量。**

## 四张核心图

1. [GPU时间线](figures/01_gpu_timeline.png)：区分观测Kernel与Graph外包络。
2. [DRAM Read时间序列](figures/02_dram_read_timeseries.png)：显示为1 ms分箱，原始采样为10 kHz。
3. [SMs Active / Tensor Active](figures/03_sm_tensor_timeseries.png)：设备级指标，不是可用资源比例。
4. [Decode时间分解](figures/04_decode_time_breakdown.png)：Profiling分解与正常基线分别作图。

每张图同时提供SVG。Graph级Trace不用于计算精确Kernel Coverage；Graph分解来自独立的32步Node采集。

## 数据和来源

```text
final_report.md / environment_check_report.md
baseline.csv / commands.txt / environment.txt / input_token_ids.json
raw/*.json                 原始请求、逐步调度、输出计数与Replay记录
logs/                      启动/采集/导出/监督日志及错误修正记录
analysis/
  metric_summary.csv       全窗与活跃/间隔采样统计
  profiler_overhead.csv    无Profiling / timeline / metrics对照
  decode_step_stats.csv    逐步输出进度延迟
  timestamp_validation.json / quality_audit.json
  metric_samples.csv.gz    完整目标指标采样CSV
  kernel_intervals.csv.gz  完整目标Kernel区间CSV
  report_data.json.gz      统计与绘图的完整中间数据
  server_trace_manifest.json
publication_manifest.json  已提交原始产物的字节数和SHA256
src/original/              实验时使用的原始GPU与分析脚本
tools/                     仓库路径适配的CPU校验/重绘工具
```

`raw/`同时保留同步调度诊断组和pilot；正式数据以
`protocol=default_scheduler_v2`识别。r01同步调度基线没有混入`baseline.csv`。
Graph pilot的自身进程重托管误报及其修正见完整报告；原始日志没有删改。

原始 `.nsys-rep` 和 SQLite 共52个文件，约743 MB，继续保存在原实验服务器
`~/fp4-llm/results/nsys_round2/`。仓库提供
[完整文件名、大小和SHA256](analysis/server_trace_manifest.json)，没有将这些二进制、模型权重或认证文件加入Git。
拥有原服务器访问权时，可用SCP取回到自己的数据目录，例如：

```bash
scp '<your-ssh-alias>:fp4-llm/results/nsys_round2/raw/<trace-name>.nsys-rep' ./
scp '<your-ssh-alias>:fp4-llm/results/nsys_round2/sqlite/<trace-name>.sqlite' ./
```

压缩的CSV和JSON为无损压缩；未增加模拟数据或丢弃采样行。
`publication_manifest.json`保留解压后的大小与SHA256。原始快照禁用Git换行转换，便于跨平台核验。

## CPU侧核验与重绘

以下命令均不启动CUDA任务，可从仓库根目录运行：

```bash
python nsys_round2/tools/verify_bundle.py
python -m pip install -r nsys_round2/requirements-analysis.txt
python nsys_round2/tools/make_figures.py
python nsys_round2/tools/make_report.py
```

后两条默认输出到`nsys_round2/rebuilt/`，保留提交的原始报告与图。
可以用`--data-root`指定本目录、`--output`指定输出位置。
跨平台字体可能影响图片字节；数值来源是同一份已校验数据。
查看完整采样时可用`gzip`解压；例如：

```bash
gzip -dk nsys_round2/analysis/metric_samples.csv.gz
```

## 重新执行GPU实验

GPU复现依赖既有 `~/fp4-llm/` 环境及模型，详见
[环境检查](environment_check_report.md)、[实际命令](commands.txt) 和完整报告。
本轮使用 vLLM 0.33.0、torch 2.14.0+cu132、FlashInfer 0.7.2.post1、
NSYS 2026.1.3.425；不沿用仓库早期实验的torch版本。

将`src/original/`中的运行与校验脚本放入该项目的`scripts/`，并保留已有`env.sh`、
`models/Qwen3-8B-BF16`与`models/Qwen3-8B-NVFP4`。
在有授权且GPU空闲的窗口内，先执行受监督的小规模采集：

```bash
cd ~/fp4-llm
source env.sh
python3 scripts/run_one.py --precision nvfp4 --mode graph \
  --kind pilot --batch 8 --steps 32 --repeat 10 --timeout 210
```

`run_one.py`包含忙闲检查、其他计算进程检测与硬超时；如果原服务器上已有同名输出，会拒绝覆盖。
共享机器上的测试仍需遵守实际授权窗口，不能仅依赖脚本判断。
`run_plan.py`记录正式矩阵的启动方式；重新运行时需要新的重复编号或新的结果目录。

原始脚本的SHA256见`analysis/report_data.json.gz`；`tools/`仅做CPU路径和压缩输入适配，
没有修改被测模型、Kernel或实验数字。
