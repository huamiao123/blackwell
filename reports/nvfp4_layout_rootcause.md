# NVFP4 GEMM 数值错误：根因分析与验收准备

日期：2026-11-14　|　状态：**根因已定位，CPU 侧证据齐全；GPU 六阶段验收待批准窗口**

## 1. 现象

`torch._scaled_mm(a_fp4, w.t(), scale_a=..., scale_b=..., out_dtype=bfloat16)`
在 RTX 5090 上**能执行、不报错、输出形状正确**，但与"用同一份 payload/scale 位
计算的 NumPy FP32 参考"相比，relative L2 ≈ **0.56**。

## 2. 被逐一排除的假设

| 假设 | 排除方式 | 结论 |
|---|---|---|
| FP4 低/高 nibble 顺序错 | 源码 `flex_gemm.py:184` 明确 `packed=(code[k+1]<<4)|code[k]`；两种顺序都试过，误差相同 | 排除 |
| 矩阵转置方向错 | one-hot 测试在 k=0/1/16/17 全部精确命中（0.000e+00） | 排除 |
| dtype 不对 | `float4_e2m1fn_x2` + `float8_e4m3fn`，与 torch 报错信息要求一致 | 排除 |
| scale 元素**数量**不对 | 2048 = 128*ceil(128/128)*ceil(16/4)*4，完全符合 | **数量对，但这是陷阱** |
| 随机数据本身有问题 | 全 1 数据也错（若全 1 正确则说明数据无关） | 见 #3 |

## 3. 根因：scale 必须是 SWIZZLE_32_4_4（L4 padding 分块），不是行主序

`torch/_meta_registrations.py:7374-7387`：

```python
padded_num_k_blocks = ceil_div(num_k_blocks, 4) * 4
expected_a_size = 128 * ceil_div(m,128) * padded_num_k_blocks
expected_b_size = 128 * ceil_div(n,128) * padded_num_k_blocks
```

`torch/_higher_order_ops/flex_gemm.py:188-216` 的 `to_blocked` 给出确切置换：

```
pad 到 (ceil(rows/128)*128, ceil(cols/4)*4)
-> reshape(rb,128,cb,4).permute(0,2,1,3)
-> reshape(-1,4,32,4).transpose(1,2)
-> flatten
```

我们之前传的是行主序 `(M, K/16)`，**size 恰好相等**（见 #4），所以 size 检查通过、
数值却错。这就是 0.56 的唯一来源。

## 4. 为什么这个 bug 不会自己暴露（关键陷阱）

```
K=256  -> kb=16  -> 128*1*16  = 2048  = 128*16     行列主序元素数相同
K=4096 -> kb=256 -> 128*1*256 = 32768 = 128*256    同样相同
```

在**本项目所有实验形状**上，元素个数都与朴素行主序重合。因此"数量对"永远不能
作为"布局对"的证据——必须逐元素比对排列。

## 5. CPU 侧证据（无 GPU，已完成）

`tests/test_to_blocked_equiv.py`：自写 NumPy `to_blocked` 与 torch 官方
`flex_gemm::to_blocked` 逐元素比对，含 padding 边界：

| shape | numel | max_abs_diff | 结果 |
|---|---|---|---|
| (128,16) | 2048 | 0.000e+00 | PASS |
| (128,256) | 32768 | 0.000e+00 | PASS |
| (128,1) | 512 | 0.000e+00 | PASS |
| (1,1) | 512 | 0.000e+00 | PASS |
| (37,5) | 1024 | 0.000e+00 | PASS |
| (129,7) | 2048 | 0.000e+00 | PASS |
| (256,3) | 1024 | 0.000e+00 | PASS |
| (100,255) | 32768 | 0.000e+00 | PASS |
| (383,257) | 99840 | 0.000e+00 | PASS |
| 索引映射 | 1024 | 0.000e+00 | PASS |

**10/10 完全一致（差值为 0，不是"在容差内"）。** 因为置换是纯重排，用整数索引
张量即可验证映射本身，不受浮点误差干扰。

## 6. 待执行的 GPU 验收（等批准窗口）

`tests/nvfp4_acceptance.py`，六阶段，每阶段输出
max_abs_error / relative_L2 / cosine / P99 / NaN-Inf / **最差元素坐标**，并落 JSON：

| 阶段 | 内容 | 判据 |
|---|---|---|
| s1 | 全 1 payload + 全 1 scale (K=256) | max_abs ≤ 1e-3（应精确 256）**门控** |
| s2 | one-hot k=0,1,16,17,K-1 | max_abs ≤ 1e-3（应精确 1.0） |
| s3 | 仅 kb=0/5/last 设 scale=2 | max_abs ≤ 1e-3（应精确 2.0） |
| s4 | 行相关 scale (row0/1/7 = 1/4/0.5) | max_abs ≤ 1e-3 |
| s5 | 随机 payload + 随机 scale | rel_L2 ≤ 2e-2（bf16 输出） |
| s6 | **同源 BF16 → NVFP4 量化** vs 量化后 FP32 参考 | rel_L2 ≤ 2e-2 |

s1 失败即中止正式验收（诊断阶段仍可单独 `--stages s2,s3` 运行）。

## 7. 原生 MMA 证明（GPU，待跑）

`scripts/prove_native_mma.py` 用 `torch.profiler` 抓真实执行的 CUDA kernel 名；
`scripts/prove_native_mma.sh` 再在拥有该符号的库里用 `cuobjdump -sass` 统计
`OMMA/HMMA/QMMA` 指令数。三条证据链：
1. profiler 显示的是 FP4 block-scaled kernel（而非 bf16 dequant 内核）；
2. 该符号的 SASS 里出现 `OMMA`（SM120 block-scaled FP4 MMA）；
3. 六阶段数值与量化参考一致。

## 8. recipe 约定（已修正并落地）

前一版把全局 scale 也四舍五入到 E4M3，导致它常落在 2^-9、块 scale 被迫顶到 448。
现已按标准做法拆成 **编码/解码两个方向**，全局 scale 保持 FP32：

```
encode:  g = 2688 / amax          (2688 = 448*6, FP32, 不做 E4M3 舍入)
         x_enc = x * g
         s_block = amax(|x_enc| in 16) / 6   -> FP8 E4M3
         q = round_e2m1(x_enc / s_block)     -> 4-bit E2M1，饱和到 ±6
decode:  x_hat = q * s_block / g
GEMM:    y = (q_a s_a)·(q_b s_b) * 1/(g_a g_b)
```

实现集中在新模块 `src/nvfp4_recipe.py`，`tests/nvfp4_acceptance.py` 的 s6 直接调用它，
不再有第二份量化代码。

CPU 侧验证（`tests/test_nvfp4_quant_recipe.py`，21 项全过）要点：
- `g = 2688/amax` 精确成立；且**证明 g 不在 E4M3 网格上**（672 → 最近 E4M3 是 448），
  这条直接证明"全局 scale 不能被压成 E4M3"。
- 编码后 `max|x_enc| = 2688`，全局最大块恰好得到 `s_block = 448`（E4M3 上界）。
- 全零张量：payload 全 0、decode 全 0、无 NaN/Inf。
- 全零 block 得到合法的非零 E4M3 scale（2^-9），且不污染相邻 block。
- amax=6 的 block 里，E2M1 的 8 个可表示值**位精确往返**。
- 1e6 与 1e-9 同块：不产生 Inf/NaN，大值饱和到 code 6，小值下溢到 0。
- 相邻两块量级相差 1e3 时 scale 分别为 448 与 0.4375，小块仍有分辨力。
- pack/unpack 往返一致；alpha = 1/(g_a g_b) 有限。

参考自洽性（`tests/test_stage6_reference.py`）：`ref_mm(...)`*alpha 与
`dequant_a @ dequant_b.T` 在三种形状上相对误差 < 1e-6，全零得精确 0，
单点热元素精确落在 (3,5)。**保证 s6 的阈值衡量的是 kernel，而不是我们自己的记账错误。**
