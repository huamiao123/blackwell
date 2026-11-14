# ⚠️ 这里的 v2 脚本已被 audit_v3 取代

本目录下的 `interference_bench.py`、`interference_v2.py`、`interference_full.py`、
`overlap_estimate.py`、`overlap_probe.py`、`bench_precision.py` **保留仅供追溯**，
它们包含已在 [`../audit_v3/source_audit.md`](../audit_v3/source_audit.md) 中记录的致命缺陷：

| 缺陷 | 位置 | 后果 |
|---|---|---|
| A1 CPU 提交偏差 | `interference_v2.py:71-89` | 两任务设备端起跑相差 **4.5–23 ms**（实测） |
| A2 串行基线是加出来的 | `interference_v2.py:110` | `G` 的分子不是实测值 |
| A3 探针不纯 | `interference_v2.py:32-44` | FP32 探针实为访存受限；DRAM 探针带归约；tensor 探针未验证 kernel |
| A4 `overlap` 命名 | `overlap_estimate.py:3` | 墙钟派生量被当作物理重叠率 |

**请勿用这些脚本的结果作为性能结论。** 修正版在 `audit_v3/src/`：

- `probe_fp32_fma.py` / `probe_dram.py` / `probe_tensor.py` — 重写的探针
- `concurrent_graph.py` — CUDA Graph 并发协议（含起跑偏差测量）
- `interference_v3.py` — 新编排（fixed_work 与 fixed_duration 分离）
- `timeline_analysis.py` — 从 GPU 时间线算物理 kernel 重叠

仍然有效的部分：`nvfp4_recipe.py`（量化 recipe，六阶段验收 12/12 通过）、
`data/nvfp4_acceptance_cloud.json`、`data/native_mma_proof.log`。
