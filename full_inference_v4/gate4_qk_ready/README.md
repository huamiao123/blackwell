# Qwen3 Decode 中有真实消费者的 Q/K 并发

在真实 E2 自回归中，对比原始 Graph、保留顺序的克隆，以及仅放开 Q 与 K 的 RMSNorm/RoPE 顺序的克隆。所有计算来自原模型。这项实验检验相互独立的短 Kernel 是否值得并发，并将 GPU 观测与无 Profiling 整批完成时间比较。[实验报告](final_report.md)

## 工作量与依赖

- RTX 5090，单卡、单模型进程。Batch 8，八条不同的 2,048-token 受控输入，257 个实际输出，256 个 Decode 步，36 层；BF16 与原生 NVFP4 共享 FP8 E4M3 KV。输入、实际输出逐请求 Hash 与本精度的历史基线核对。测量从预分词后的本地请求进入到返回。
- 原始顺序为 QKV → Q 分支 → K 分支 → KV 写入。第三种配置在独立克隆上保留 QKV → Q 和 K → KV 写入，增加 QKV → K 与 Q → KV 写入，移除 Q → K。Q/K 两个结果均就绪后再交还原 Graph。未增加 Kernel、复制算子或改变其参数。
- 36 层分别用真实模块 Q/K norm 权重地址识别。实际 Graph 的参数中，Q/K 共享 QKV 输入、positions 和 cos/sin 表。核验两个输出的地址范围与彼此及所有输入区间互斥。K 的输出地址直接出现在原 KV 写入 Kernel 的首个参数中；Q 的输出地址实际出现在 Attention 参数中。缓存写入到 Attention 的原边及边属性完整保留。Q/K 参数边仅接受普通依赖。生成代码审计详情见 [本机源文件与内存读写核查](analysis/source_audit.json)。
- P2 按预先固定的顺序，在同一完整推理的 256 个实际 Decode 步交错运行三种版本，以 NVTX → CUDA launch correlation → exec ID → local node ID → 完整 Kernel 签名核验每层每步。P0 在独立进程执行，每次完整推理的全部 256 步只使用一种版本。各进程五个平衡块，块内三种版本按固定种子随机排序。两种精度交替执行，每精度、每 Profiling 条件均独立进程复测。每次 P0 和 P2 都进行 64 个暖机 Replay。
- 推理内容使用贪心自回归，未更换调度器、Graph 捕获形状或 Attention 后端。工作进程在本机现有安全监督器下运行，按次检查共享设备和外部进程，遇冲突只终止本任务，单次硬超时 250 秒。实验进程内的温度、时钟与功耗区间来自同一监督器。[遥测](analysis/telemetry.csv)

## CPU 重建

从仓库根目录执行以下命令，以已保存的推理记录与 GPU 区间重算时间和统计。绘图使用 matplotlib；分析使用 Python 标准库。

```sh
python full_inference_v4/gate4_qk_ready/src/analyze.py
python -m unittest discover -s full_inference_v4/gate4_qk_ready/tests
python full_inference_v4/gate4_qk_ready/src/plot.py
```

每个成功 P2 的原始报告及 SQLite 位于实验服务器；目录中保留路径、尺寸和 SHA256。对已保存的 Trace 重新执行分析可使用 `python full_inference_v4/gate4_qk_ready/src/get_trace.py HOST RUN_ID`。已冻结的配置、上轮精度内输出哈希、源代码版本和已有模型缓存是复核输入。每个成功工作进程均来自同一份修订后的实现。[运行索引](analysis/inventory.csv) · [实际源码修订](analysis/worker_revisions.json)

## 关键统计口径

| 指标 | 定义 |
|---|---|
| 时间重叠 | max(0, min(Q结束, K结束) − max(Q开始, K开始)) |
| Q/K 联合区间 | 从 Q/K 首个开始，到最后一个结束的 GPU 时间 |
| 到达消费者前的时间 | 从 QKV 结束到原 KV 写 Kernel 开始，包含进入和汇合开销 |
| 模型时间 | 本地 E2 八请求生成返回，截止时间在统一 drain 之前 |
| TPOT / ITL | 请求级平均每输出 Token 时间，及各请求相邻输出事件的间隔。P95 在每个 trial 内计算 |
| P0 干扰 | 同块 fork / 顺序克隆 − 1，逐进程报告均值与五块的描述性 95% t 区间 |
| 理想节约 | 顺序克隆每层联合区间减两分支中较长的实际时长，36 层求和，再按每步中位数乘 256；假定放开依赖后执行时长与其他成本均保持原值 |

首个 NVFP4 P2 在暖机阶段发现原图 KV 写入到 Attention 有专用依赖属性，主动停在拓扑审查处，未产生测量。修订后只记录这条原边的属性，完整保留其语义；修改的三条 Q/K 边均为普通依赖。失败记录保留在相同索引中。数值一致性检验覆盖输出 Token；中间浮点缓冲区未逐元素导出。P2 混排而 P0 按请求固定条件，因此没有逐配置、完全相同调度的 profiler 开销测量。[含首次失败的监督记录](analysis/supervision.csv)
