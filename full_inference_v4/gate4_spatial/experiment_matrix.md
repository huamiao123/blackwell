# 冻结实验矩阵

初始主节点：layer 18 的 BF16/NVFP4 QKV 和 Gate/Up。第一试点进一步收窄到 QKV + FP32 Light，Grid=32、128；五个配对 Block。Gate/Up、Tensor Probe、Heavy 和其余 Grid 不自动执行。

| 条件 | 用途 |
| --- | --- |
| A Alone | 本轮实际主节点独立基线 |
| B Alone | 同一配置与完整工作量的独立辅助基线 |
| Serial AB / BA | 实测串行与顺序影响 |
| Parallel AB / BA | 不同 Stream，测真实起跑、重叠和竞争 |
| Empty Control | 辅助提交和空 Graph 本身的开销 |
| Same Stream Control | 强制同 Stream 的提交负对照 |
| Non-overlap Control | 不同 Stream、明确结束依赖的负对照 |

候选独立 Grid 扫描为 16、32、64、128、170、192、256，不执行完整笛卡尔积。Grid 变化使用相同总逻辑 items/tiles、迭代数和输出；单 CTA 工作量会相应变化，必须报告其影响。Light/Heavy 的实现与资源配置分别建基线，不将它们混成同一个 Kernel。

`configs/experiment_plan.json` 是预先生成的计划；`PLANNED_NOT_RUN` 行不是已运行观测。计划含 20 个 condition-block（每个含九条件），不是 20 次独立进程，也不证明统计独立。

只有正确性、负对照和 P2 映射通过人工审查后才扩大试验。O/Down 单列为后续方向，没有自动执行入口。
