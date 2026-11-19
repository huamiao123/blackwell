# 完整 Decode 中的 O 与 Down 投影放置对照

本目录检验一个可证伪的预测。在相同 E2 推理的第 18 层，把同一 BF16 Tensor 探针放到 O 或 FFN Down 的并行分支，测量它能否在投影结束前做完，并检验差异能否传到完整请求。预测是实验前按旧时间线和独立探针时长算出的。实验结果、适用范围和证据等级见[研究报告](final_report.md)。

## 数据与进入条件

沿用 E2 的本地 vLLM 自回归执行、实物 RTX 5090、两个精度各自的真实 checkpoint、8 条不同的 2,048-token 输入、257 个输出、256 次 Decode Replay、FP8 E4M3 KV、关闭前缀缓存。逐请求、逐精度核对了旧 E2 的输入和输出 Hash。输入来自受控 Token，测量从已分词的本地请求到模型响应。在线服务延迟需要另行测量。[E2 配置](../configs/) · [基准与先验预测来源](../gate4_microarch/final_report.md)

`src/target_graph.py` 在实例化期间克隆捕获的完整 Graph，用第 18 层现场模块的权重地址、NVFP4 scale/alpha 地址、参数大小、完整函数符号、Grid/Block 和共享内存识别投影。原 Graph 用于原生对照。独立的可执行 Graph 复用活体模型缓冲区及原始流，探针输出存放在逐步独立缓冲区。实测区间前完成实例化和上传。P0 另以两组小输出预热请求覆盖各版本，共计 64 次预热 Replay。[实现](src/target_graph.py) · [CUDA Driver Graph API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)

| 条件 | GPU 依赖和有效工作 |
|---|---|
| original | 重放原 Graph，无 B |
| sham | 重放完整 Graph 的克隆，无 B |
| o_parallel / down_parallel | A 和 B 继承 A 的直接前驱，A 的所有直接后继等待两者完成 |
| o_ab / down_ab | B 在 A 完成后进入，A 的直接后继等待 B |
| o_ba / down_ba | B 继承 A 的前驱，A 等待 B，原后继依赖保留 |

A 是指定投影，B 是固定的 64 CTA BF16 Tensor MMA 探针。每 CTA 有 32 个线程和 1,024 B 动态共享内存。每次注入完成 268,435,456 逻辑 FLOP，校验全部 65,536 个输出元素。每次测量前都有独立探针正确性与耗时检查。GPU 时间线计算 A/B 执行交集、B 被覆盖的时间占比、B 延迟进入量和结束余量。独立 preflight 的 CUDA Event 时间与 GPU 单 Kernel 执行时间使用各自的分母。[探针实现与编译清单](../gate4_spatial/analysis/offline_compile/) · [Trace 关联与断言](src/trace.py)

## 重建分析

从仓库根目录运行以下 CPU 命令。图表需要 Python 的 matplotlib。分析器自动使用已提交的原始压缩数据、旧配置和历史逐请求 Hash。统计保留每个独立进程和每个配对块。

```sh
python -m unittest discover -s full_inference_v4/gate4_targeted/tests
python full_inference_v4/gate4_targeted/src/analysis.py
python full_inference_v4/gate4_targeted/src/figures.py
```

服务器上每个 P2 的完整 SQLite 与 .nsys-rep 均有路径、字节数与 SHA256 清单。对已有 Trace 重新抽取节点以及采集已保存的监督遥测可以通过以下 CPU 命令完成。HOST 是已配置的实验服务器 SSH 别名，RUN_ID 取自运行清单。

```sh
python full_inference_v4/gate4_targeted/src/collect_trace.py HOST RUN_ID
python full_inference_v4/gate4_targeted/src/collect_meta.py HOST
```

在有效的共享 GPU 窗口内，`src/submit.py HOST nvfp4 P2 r1` 只提交一个有监督、具有时限的配置。参数可取 bf16/nvfp4、P0/P2、r1/r2。使用已有 E2 模型、依赖和数据。P2 的 r1、r2 和 P0 的 r1、r2 使用固定的两个排序种子 71011、84239。每个 P2 的 32 步分四个平衡块，每块八种模式各出现一次。每个 P0 进程是五个随机顺序的平衡块，块内每种模式独立跑一次完整推理。同一模式在该次推理的第 33–64 个 Decode 步使用，其余步骤使用原 Graph。[运行清单](analysis/run_inventory.csv) · [监督与遥测](analysis/supervision.csv)

## 结果文件

| 文件 | 可回答的问题 |
|---|---|
| [逐步 GPU 记录](analysis/p2_nodes.csv) | 哪一步实际运行哪个 exec，A/B 何时开始和结束，何时重叠 |
| [时间线汇总](analysis/p2_summary.csv) | 分进程的探针覆盖、进入延迟、结束余量与局部串并行时长 |
| [事前估计检验](analysis/forecast_check.csv) | 哪些投影能容纳该探针，观测与原估计相差多少 |
| [无 Profiling 原始计时](analysis/p0_trials.csv) | 每次完整请求组的模型与联合耗时、注入步 ITL、TTFT 与 TPOT |
| [逐块差值](analysis/paired_blocks.csv) / [进程内估计](analysis/p0_summary.csv) | 相同工作量串并行收益、Down 与 O 的成对差异和 95% 区间 |
| [原始记录](raw/) / [工作进程源码版本](analysis/source_versions.json) | 复核输出、Graph 身份、源码修订及服务器上的原始 Trace |

首份 NVFP4 P2 的 NSYS `stop-shutdown` 导致监督命令返回早于 Python 写完最终结果。随后从原进程的完整输出恢复了原始记录，核对了 256 次 Replay、输出、探针和 Trace。后续 P2 改为 `stop`，等待进程完整退出。四份 Trace 都核验了每次 Replay 的原模型 Kernel 符号和资源配置多重集。后三份另逐 local node ID 核对完整签名。首份的目标节点也通过精确权重绑定、exec ID、节点 ID、符号和形状校验。每次实际部署的源文件哈希及较早版本差异均有留档。
