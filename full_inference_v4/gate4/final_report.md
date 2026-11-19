# Gate 4 阶段报告

**记录日期** 2026-11-19

## 机理阶段进展

| 问题 | 结果 | 证据等级 |
| --- | --- | --- |
| 两种精度能否与额外计算在 GPU 上重叠 | QKV 和 Gate/Up 的指定组合均观察到真实 GPU Kernel 时间重叠，分属同 Context 的独立 Stream | `VERIFIED`，仅限实际测量条件 |
| NVFP4 是否一概更容易容纳额外计算 | 一组 BF16 Tensor Light 在 NVFP4 QKV 期间保留更多辅助吞吐。Gate/Up 的 FP32 Light 反例表明不能从精度和 Grid 单独决定并发策略 | `VERIFIED`，有配对 P0 和独立 P2 |
| Grid 变化影响什么 | QKV FP32 Light 的 64 CTA 比 128 CTA 具有更长的 B 尾部。相同工作量下 NVFP4 的联合时间约多 16 μs，部分顺序主干扰降低 | `VERIFIED`，对应固定工作量范围 |
| NVFP4 重 CTA 会否限制同 SM 共驻 | 数个配置超出混合资源必要上界。QKV + FP32 Heavy 仅 Dynamic SMEM 即超限，同时观察到 GPU 区间重叠 | `PARTIALLY_SUPPORTED`，支持跨 SM 解释 |
| 是否直接验证同 SM 共驻 | 当前采集没有 CTA 级 SM 位置 | `UNVERIFIED` |
| 能否改善完整推理 | 单投影伴随 BF16 MMA 探针可完成额外工作，Prefill、Decode、请求生命周期内的真实消费者和服务净收益待验证 | `UNVERIFIED` |

[P0、P2 数值和失败记录](../gate4_spatial/qkv_replay_validation/qkv_gateup_20261119.md) · [模型与实测驻留关联](../gate4_residency/analysis/live_overlap_evidence_20261119.csv)

## 软件与数据完整性

逐层资源表由已有 Gate 3 原始表自动关联 288 条记录，记录 Graph、Node、CUDA Tools ID、各维 Grid/Block 和占用上界。CPU 侧覆盖 28 项正确性检查。六种 DRAM Streaming 变体完成 SM120 离线编译和访存指令审核，实际 DRAM 性能及正确性待 GPU 验收。历史 Gate 2/3 文件保留原内容。

场景限定于稳定输入的独立 CUDA Graph。原始源码、真实执行、PyTorch Graph 捕获与输出一致性均经过核对。全模型调度、全权重工作集和混合请求属于另一个适用范围。单机配对置信区间只描述当前设备和运行协议。

## 研究推进边界

任务一的优先矩阵覆盖 QKV 和 Gate/Up，2026-11-19 共新增四组六块 P0 配对和八份独立 P2 Trace。O、Down 的扩展有完整历史映射，尚缺独立重放及并发数据。任务二已有活函数资源、同质 Occupancy API 和混合资源与时间线对照，实际 CTA 位置尚无直接证据。

任务三的阶段注入、调度及有用工作定义已有设计。短 BF16 MMA 是当前最有证据的探针。转入真实完整推理时需以实际下游消费为准，尤其验证可用输入、增量成本、结果截止时间、缓存干扰和请求 P95。完整 E2 Prefill/Decode/生命周期并发仍未执行。[完整推理候选审计](../gate4_inference/candidate_audit.md)
