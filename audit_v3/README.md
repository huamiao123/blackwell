# audit_v3 — 对 V2 干扰实验的审计、修正与重新验证

本目录不覆盖任何旧结果。旧实验保留在仓库根目录（`src/`、`data/`、`reports/M1_*`），
其问题记录在 [`source_audit.md`](source_audit.md)。

## 一句话结论

**旧的并发协议存在致命缺陷（两个任务实际起跑相差 4.5–23 ms），因此
所有 `S_A`/`S_B`/`G` 的绝对值不可用、`overlap %` 不是物理重叠率。
改用 CUDA Graph 协议后，唯一经受住修正的定量结论是：
M=1024 时 BF16 主任务被 tensor 干扰只慢 4.8%，而 FP4 被慢 57%。**

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
| 1 最小复现（3 M × 2 精度 × tensor 探针 × k=1） | **完成** | 见 corrected_results.md |
| 2 三类探针验证 | 未开始 | FP32/DRAM 探针代码已重写，尚未运行验证 |
| 3 全 M 重测 | 未开始 | 需 Gate 2 通过 |
| 4 机制验证（CTA 限制共驻实验） | 未开始 | 见 hypotheses.md H3 |

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
