# Gate 4 后续对照与当前验收范围

已完成 QKV 的 BF16/NVFP4 单节点回放、实际 Occupancy API、32/128 CTA FP32 Light 的 P2、六个 P0 配对块与尾部解释。64 CTA、Gate/Up 与新辅助任务组合已经接入同一受监督入口，尚待 GPU 实测。

按下列顺序推进，前一组输出或时间线有错误时定位后再扩展。

| 顺序 | 测试 | 固定条件与通过标准 |
| ---: | --- | --- |
| 1 | QKV 64 CTA FP32 Light | 两种精度各一份 P2，通过九条件输出与负对照。随后做 64/128 同进程六配对块。保持 65536 items、8 轮和 16777216 FLOP，从实际 GPU 区间统计辅助尾部。 |
| 2 | Gate/Up 独立重放，随后与 FP32 Light 配对 | 只选第 18 层。从历史 Gate 3 验证 Gate/Up 的真实符号、192/1536 CTA、参数大小、权重引用及运行时资源。输出必须与该精度 vLLM 原路径逐元素一致，再做小 Grid 的 P2。 |
| 3 | QKV FP32 Light/Heavy 资源对照 | 先运行各自 B Alone 检查该短任务全部输出和 live 属性。相同 Grid 和相同 FMA 工作量，再测实际负对照、干扰、并发与 tail。Heavy staging 实际读写。 |
| 4 | QKV BF16 Tensor Light | 固定 256 个 16×16 输出 tile，128 轮 BF16 K=16 MMA。先核对独立输出、时长和真实指令属性，确认时长后再审查正式矩阵。 |
| 5 | 有真实消费者的完整推理试点 | 先验证原执行路径的真实必要工作、可提前依赖及实际消费者。候选使用独立请求的 BF16 LM Head，须另外建立实际资源竞争的证据和合法调度。 |

Gate/Up 分支复用 QKV 的 Loader，按 vLLM 融合顺序读取 gate、up shard。NVFP4 使用已验收的原始 CUTLASS 权重处理，BF16 使用同一 `dispatch_unquantized_gemm`。数值及实际 Graph 身份验收前，历史指纹只作为约束。新探针复用已在 RTX 5090 验收过的 CUBIN，但新的短工作量仍需 B Alone 输出校验。历史结果保留，分析器复算已归档的 32 CTA 时间表结果一致。

23:03 的检查仍发现外部计算进程和自动实验控制器。当前只完成上述入口、源代码审查和已有结果分析。重新使用共享 GPU 前必须通过同样的进程和控制器检查。

- [实机 CTA 属性的驻留模型](../../gate4_residency/live_qkv_report.md)
- [真实消费者与 Amdahl 候选筛选](../../gate4_inference/candidate_audit.md)
