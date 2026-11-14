# 源码审计：V2 干扰实验（`src/interference_v2.py` 及其下游）

审计对象：`src/interference_v2.py`、`src/interference_full.py`、`src/overlap_estimate.py`、
`src/overlap_probe.py`、`src/bench_precision.py`、`src/make_figures.py`、`src/nvfp4_recipe.py`、
`tests/nvfp4_acceptance.py`、`data/interference_full*.json`、`reports/M1_resource_difference_first.md`。

判定口径：
**FATAL** = 可能推翻既有结论；**MAJOR** = 系统性偏差，需重测；**MINOR** = 影响精度但不改结论。

---

## A1【FATAL】CPU 提交顺序造成的起跑偏差未被控制

**位置**：`src/interference_v2.py:71-89`（`concurrent()`）

```python
ready.record(ctrl)
sA.wait_event(ready); sB.wait_event(ready)
with torch.cuda.stream(sA):
    a0.record(sA)
    for _ in range(rA): fA()        # 先全部提交 A
    a1.record(sA)
with torch.cuda.stream(sB):
    b0.record(sB)
    for _ in range(rB): fB()        # 再提交 B
    b1.record(sB)
```

`ready` 事件只保证两条流"不早于 ready 开始"，**不保证同时开始**。
主机按顺序把 rA 个 A kernel 压进队列（rA 实测可达 867），再压 rB 个 B kernel
（rB 可达 353）。GPU 在主机压 B 之前就已经开始跑 A。

**证据（同一份 JSON 里自证）**：`data/interference_full.json` 中大量条件下
`T_A_ms ≈ T_joint_ms`：

| M | dtype | probe | k | T_A | T_B | T_joint |
|---:|---|---|---|---:|---:|---:|
| 128 | fp4 | fp32 | 1.0 | 30.98 | 16.54 | 31.03 |
| 128 | bf16 | fp32 | 1.0 | 36.51 | 20.63 | 36.54 |
| 1024 | bf16 | dram | 1.0 | 37.32 | 26.91 | 37.35 |

若两者真同起跑，joint 应 ≈ max(T_A,T_B)（此表恰是如此），但 T_A 本身已在并发中被拉长
（S_A≈2.0），说明 **A 先跑、B 后到、B 只在 A 的尾部与下一轮重叠**。这使
`S_A` 的物理含义变成"A 与 B 尾部相撞时的减速"，而不是"两任务全程竞争"。

**是否推翻结论**：部分推翻。修正后实测差异只有 **−0.21…+0.04**（见 corrected_results §1），
**不是** ±0.63。先前那个"±0.63"的估计来自 A12 的工作负载 bug，不是提交协议本身。
起跑偏差本身仍然真实且很大（1.0–14.6 ms），但它对 S_A 的影响是有限的。

**修复**：两条任务各自预构建 CUDA Graph，用同一 latch event 触发，并分别测试
A→B、B→A 两种提交顺序；同时量化 CPU 提交耗时（`concurrent_graph.py`）。
**验证**：修正前后对比 `T_A_conc - T_A_alone` 与 `T_joint - max(T_A,T_B)`；
若提交偏差消失，二者应接近 0。

---

## A2【FATAL】串行基线不是测出来的，是加出来的

**位置**：`src/interference_v2.py:110`

```python
tSer = timed(fA, rA) + tB      # 两次独立测量相加
```

V2 任务书 §6.3 明确"**严禁使用 (T_A+T_B)/T_joint 代替；必须使用实验测量的 T_serial**"。
当前 `G = T_serial/T_joint` 的分母是实测、分子是两次独立运行之和，
它衡量的是"并发相对理想串行的收益"，而不是"相对真实串行的收益"。
更糟的是：`timed(fA,rA)` 与 `tB` 在不同时刻测，缓存状态不同，相加无物理对应。

**是否推翻结论**：会。`G` 的全部数值（0.96–1.25）不可用。

**修复**：在同一会话内实测 `A→B` 与 `B→A` 两种串行顺序，取两者均值/各自报告。

---

## A3【FATAL】三类探针的设计与其名义资源不符

| 探针 | 位置 | 实际行为 | 问题 |
|---|---|---|---|
| `fp32` | `interference_v2.py:32-35` `torch.add(x,y,out=x)` | 128 MB 逐元素加法 | **访存受限**，不是 FMA 流水线压力；无法代表 FP32 计算 |
| `dram` | `:37-40` `torch.sum(x)` | 1.5 GB 归约 | 归约树 + 临时量 + 内建 kernel，算术强度非零；且无法区分 L2 复用与 DRAM |
| `tensor` | `:42-44` `a @ b`（2048³ bf16） | cuBLAS GEMM | 未验证 kernel 身份/是否用 TC；未做 FP4 对照；网格大小未控制 |

**是否推翻结论**：会。"FP32/DRAM 探针几乎不影响主任务（重叠 3–4%）"这一条，
可能只是"这些探针本身是访存受限、与 GEMM 抢的不是同一条通路"，而非"FP32/DRAM 无余量"。

**修复**：Triton 写真正的 FMA 循环（多独立累加器、寄存器内、PTX 里验证 `fma.rn.f32`）；
DRAM 用连续合并读写 + 工作集扫描；Tensor 探针同时提供 BF16 与 FP4 两种并验证 kernel 名。

---

## A4【MAJOR】`overlap` 命名把墙钟推算伪装成物理重叠

**位置**：`src/overlap_estimate.py:3-8`、`src/make_figures.py:5,9,31,33,34`

```python
def ov(tA,tB,tJ):
    hi,lo = max(tA,tB), min(tA,tB)
    return max(0,min(1, 1-(tJ-hi)/lo))
```

这是"相对串行的时间节省比例"，不是 kernel 时间区间交集。把它命名为 `overlap`
并写进图标题 `Concurrent overlap (wall-clock estimate)` 是误导。

**修复**：重命名 `serial_work_saved_fraction`（SWSF）；
物理重叠只从 GPU 时间线（profiler trace / nsys）计算，并单独命名为 `kernel_overlap`。

---

## A5【MAJOR】单独基线只在 k 循环外测一次，且不随会话重复

**位置**：`src/interference_v2.py:101-102`

```python
tA1 = single_ms(fA)          # 5 次调用估一个值
rA  = max(5, round(20.0/tA1))
tA  = timed(fA, rA)          # 只测一次，之后 3 个会话、9 个条件全靠它
```

`S_A = ca/tA` 的分母是**一次**测量。若该次落在低频/预热不足窗口，全部 S_A 同比例偏移。

**修复**：每个 (dtype, M) 在每个会话内重新测 alone，并与并发条件交错（A/B/A/B）。

---

## A6【MAJOR】条件顺序固定，未随机化

`for M → for dtype(bf16,fp4) → for probe → for k` 全固定。BF16 恒在 FP4 之前、fp32 恒在前。
GPU 温升、时钟爬坡、L2 残留都会系统性偏向后面的条件。

**修复**：条件列表打乱（固定 seed 记录），每会话重打乱一次。

---

## A7【MODERATE】JSON 存的是"比率的中位数"，破坏代数一致性

**位置**：`interference_v2.py:117-121`

`S_A` 存的是 3 个会话 `S_A` 的中位数，而 `T_A_ms` 是 3 个会话 `T_A` 的中位数。
二者不满足 `S_A = T_A_conc/T_A_alone`，下游无法反推，也无法给置信区间。

**修复**：保存每个会话的**原始样本**（T_A_alone、T_A_conc、T_B_alone、T_B_conc、T_serial、
T_joint、功率、时钟），派生指标由分析脚本从原始样本计算。

---

## A8【MODERATE】功率/时钟只在并发前采一次

**位置**：`interference_v2.py` 中 `st = gpu_state()` 一次采样。
报告中"184–330 W、无功率墙"的结论基于瞬时点采样，不是窗口平均。

**修复**：并发窗口内以 50 ms 周期采样，报告 max/mean 与限频原因（`clocks_throttle_reasons`）。

---

## A9【MINOR】压力档 k 是标称值，实际 T_B/T_A 未记录

`rB = round(k*tA/tB1)`，但 `tB1` 来自 5 次平均，且并发时 B 被拉长。
JSON 里没有 `T_B_alone/T_A_alone` 的实测比值。

**修复**：记录实测 `T_B_alone/T_A_alone`，报告标称 k 与实测比值之差。

---

## A10【MINOR】`nvfp4_recipe.py` 的 E2M1 舍入未按 round-to-nearest-even

`_codes_from_mag` 用 `a-lo <= hi-a` 取低值（ties→lower）。PTX `cvt.rn.satfinite.e2m1x2.f32`
是 RN（ties to even）。仅在恰好中点的输入上有差异，验收测试未覆盖该点。

**修复**：补 tie 用例；若需与硬件逐位一致，改为 even 规则。

---

## A12【FATAL，评审发现】legacy 分支根本没有执行主任务

**位置**：`audit_v3/src/interference_v3.py`（首次提交版本）

```python
if "legacy" in modes:
    fA1, fB1 = probe_factory, probe        # ← 错：probe_factory 是"创建探针"的工厂
    rec["legacy_AB"] = CG.run_concurrent_streams(fA1, fB1, rA, rB, "AB")
```

`fA` 才是 BF16/FP4 主 GEMM；`probe_factory()` 返回的是一个新的 `TensorProbe` 对象
（只分配显存，不做 GEMM）。于是 legacy 路径反复"创建探针对象"，而 `alone_A` 是主 GEMM
的耗时——**分子分母不是同一个任务**。M=1024/BF16 的 `S_A = 0.415` 就是这么产生的。

**影响**：所有旧协议对照数据（legacy 的 S_A/S_B、提交耗时对比）作废。
**不污染** Graph 模式，因为 Graph 路径用的是正确的 `fA`。
**修复**：`fA1, fB1 = fA, probe`，并重测 Gate 1。
**修复后**：旧协议 vs 新协议差异 −0.21…+0.04（不再有数量级差异）。

## A11【已确认无问题】
- `src/nvfp4_recipe.py` 的 encode/decode 分离（全局 scale FP32 不舍入）——与 torch 语义一致，
  六阶段验收 12/12 通过，`to_blocked` 10/10 逐元素为 0 差异。
- 原生 MMA 证据（单函数 SASS 256 条 `OMMA.SF`、0 条 `HMMA`、负对照 0）——独立于上述问题。
- 单任务吞吐表（`bench_precision_first.json`）——虽然未验证 kernel 名，但每次都是同一实现，
  趋势可用；绝对值的"峰值百分比"解释需按第 9 条纪律收回。

---

## 汇总

| ID | 级别 | 影响 | 是否推翻结论 |
|---|---|---|---|
| A1 | FATAL | 起跑偏差，S_A/S_B 物理含义错位 | 是 |
| A2 | FATAL | G 分子非实测 | 是 |
| A3 | FATAL | 三类探针不代表其名义资源 | 是 |
| A4 | MAJOR | 命名误导（墙钟→物理重叠） | 是（措辞层面） |
| A5 | MAJOR | alone 基线不随会话重测 | 是 |
| A6 | MAJOR | 条件顺序固定 | 部分 |
| A7 | MODERATE | 中位数比率不可反推 | 否（数据可重算） |
| A8 | MODERATE | 功率点采样 | 否 |
| A9 | MINOR | k 标称值 | 否 |
| A10 | MINOR | E2M1 tie 规则 | 否 |
| A11 | — | 数值/指令证据成立 | 保留 |

**必须撤回的结论**：
1. 报告中所有 `overlap %`（FP32 4.4% / DRAM 3.2% / tensor 36.9%）——不是物理重叠率。
2. 所有 `G` 值——分子非实测。
3. "FP32/DRAM 无余量"的强表述——探针本身不纯。
4. "M≈256 交叉"——需要在修正协议 + 纯探针下重测才能确认（当前只是未经证实的观察）。

**可以保留的结论**：
1. BF16 实测封顶 212.6 TF、FP4 封顶 1156.0 TF（同一 kernel 家族内自洽）。
2. 单任务下 FP4 比 BF16 快 1.09–5.44×，且加速比随 M 增大。
3. 原生 FP4 MMA 的 SASS 证据与六阶段数值验收。
