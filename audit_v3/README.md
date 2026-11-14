# audit_v3 — 对 V2 干扰实验的审计、修正与重新验证

本目录不覆盖任何旧结果。旧实验保留在仓库根目录（`src/`、`data/`、`reports/M1_*`），
其问题记录在 [`source_audit.md`](source_audit.md)。

## 一句话结论

**旧的并发协议存在致命缺陷（两个任务实际起跑相差 4.5–23 ms），因此
所有 `S_A`/`S_B`/`G` 的绝对值不可用、`overlap %` 不是物理重叠率。
改用 CUDA Graph 协议后，唯一经受住修正的定量结论是：
M=1024 时 BF16 主任务被 tensor 干扰只慢 4.8%，而 FP4 被慢 57%。**

> 后续复核又发现 legacy 分支跑错工作负载（L1），修复后旧协议与修正协议的差距
> 只有 −0.21…+0.04；**起跑偏差本身是真的（1.0–14.6 ms），但它对 S_A 的影响有限。**
> 完整修正见 [`corrected_results.md`](corrected_results.md) §0–§1。

## 目录

```
source_audit.md        逐项源码审计（11 项问题，含文件:行号、级别、是否推翻结论）
methodology.md         可复现的新协议
hypotheses.md          5 条可证伪假设（支持/反对/证伪实验）
corrected_results.md   已验证 / 已撤回 / 未决 三类结论
src/probe_fp32_fma.py  真正的 FP32 FMA 探针（Triton，4 条独立累加链，PTX 校验）
src/probe_dram.py      DRAM 流式探针（Triton，连续合并读写）
src/probe_tensor.py    BF16 / FP4 tensor 探针 + kernel 身份识别
src/concurrent_graph.py CUDA Graph 并发协议（含 host 提交与设备起始偏差测量）
src/interference_v3.py 实验编排（fixed_work / fixed_duration 分离）
src/timeline_analysis.py 从 GPU 时间线算物理 kernel 重叠
src/analyze_gate1.py   Gate 1 修正 vs 旧协议对比
src/timeline_capture.py 时间线采集
src/make_figures_v3.py 三张新图
tests/test_metrics_v3.py CPU 侧 16 项指标/区间代数测试
raw/                   原始 JSON 与日志
figures/               图
```

## Gate 状态

| Gate | 状态 | 说明 |
|---|---|---|
| 1 最小复现（3 M × 2 精度 × tensor 探针 × k=1） | **完成（已修正 legacy bug 重测）** | 见 corrected_results.md §1 |
| 2 三类探针验证 | **完成** | FP32 FMA PTX 校验通过、DRAM 在 96 MiB 出现 L2 台阶、tensor kernel 身份确认 |
| 3 全 M 重测 | **完成**（126 条件） | BF16 单调 1.767→1.056，FP4 平在 ~1.6，交叉点 M≈32–64 |
| 4 机制验证（CTA 限制共驻） | **完成（首轮）** | 1 CTA 探针的 kernel 重叠仍只有 24%，支持"无空闲 SM"假说 |
| 5 统计与前沿 | **完成** | bootstrap 95% CI、归一化 Pareto 图、逐 M kernel 身份表 |
| 6/7 固定 kernel 复核 | **完成** | 同一 kernel（`128x64_64x3`）跨 M=32/64/256/512，S_A 仍单调 1.576→1.201 —— 趋势不是 kernel 假象 |
| 任务一（P0）严格固定工作量 | **完成 5/7 个 M** | 同一 GEMM 次数下 FP4 组合最多快 4.5×（M=1024/tensor/L1）；L3 重载时优势消失、FP32 探针下反转。见 `task1_fixed_work.md` |
| 算力 vs 带宽分离 | **完成（有混杂）** | 纯 DRAM 竞争两者相同（~2.0）；FP4 的额外敏感度取决于辅助任务的网格规模，见 `corrected_results.md` §9 |

### 新增（第二轮补充）

- **`acceptance_answers.md`** — 逐条回答原任务的 9 个验收问题
- **bootstrap CI**（`raw/gate5_ci.csv`）：S_A 的 95% CI 宽度约 ±0.01
- **新发现的严重混杂**：BF16 主任务的 kernel **随 M 变化**（M=16 是 WMMA，
  M=128/1024 是不同 tile），FP4 恒定。因此 BF16 的 S_A–M 曲线**不能**直接归因于 M，
  必须在固定 kernel 后重测（见 `corrected_results.md` §7）
- **M=128 提交顺序不对称复测**（5 会话）：A→B 恒为 ~0.46 ms，B→A ~0.03 ms，
  但两种顺序下 S_A 完全相同（1.562–1.564），说明该延迟不改变竞争格局

### 评审后的修正（v2 → v3 数据）

| 问题 | 处置 |
|---|---|
| **L1** legacy 分支跑错工作负载（`probe_factory` 当主任务） | 已修，legacy 全部重测；旧"±0.63 偏差"结论撤回，实际为 −0.21…+0.04 |
| **L2** 时间线按 kernel 数量猜 stream 身份 | 已改为"同一流上单独 profile 学 stream id"；gate4 用 kernel 名匹配 |
| **L3** 起跑偏差只报合并中位数 | 已改为 AB / BA 分列 |
| 机制断言（时间片交替/L2 污染/SM 抢占） | 降级为假设（hypotheses.md H3） |

## 复现

```bash
cd audit_v3
python tests/test_metrics_v3.py          # CPU，无 GPU
python src/interference_v3.py --gate1 --Ms 16,128,1024 \
    --precisions bf16,fp4 --probes tensor --ks 1.0 \
    --sessions 3 --modes graph,legacy --json raw/gate1_v2.json
python src/analyze_gate1.py raw/gate1_v2.json
python src/timeline_capture.py
python src/make_figures_v3.py
python src/gate2_probes.py          # 探针自检
python src/gate4_coresidency.py     # 受限 CTA 共驻实验
python src/gate5_stats.py           # bootstrap CI + Pareto
python src/kernel_identity.py       # 逐 M kernel 身份
python src/analyze_gate3.py         # 全 M 汇总 + 图
```

## 图

- `figures/fig_v3_protocol.png` — 同一条件下两种协议的 S_A，以及起跑偏差（对数轴）
- `figures/fig_v3_overlap.png` — 任务区间重叠 vs 物理 kernel 重叠
- `figures/fig_v3_sA.png` — 修正协议下 S_A 随 M 的变化

## 明确不做的事

- 不修改任何旧 JSON / LOG / 报告。
- 不把墙钟派生量称作物理重叠。
- 不用"相对理论峰值的差额"解释为可回收算力。
- 不在 Gate 1/2 未通过时扩大实验。
