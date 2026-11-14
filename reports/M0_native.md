# M0 - toolchain / native instruction / quantization acceptance

Date: 2026-11-14  |  Machine: 10.60.23.250 (bare metal, 2x Xeon 8481C, 30.9 GiB RAM)
GPU: RTX 5090, cc 12.0, 170 SM, 96 MiB L2, smem/CTA opt-in 99 KiB, 64K regs/SM

## Verified facts (measured, not assumed)

| Item | Value | How |
|---|---|---|
| L2 | 100,663,296 B = 96 MiB | `cudaDeviceProp.l2CacheSize` |
| sharedMemPerBlockOptin | 101,376 B (99 KiB) | device_info |
| sharedMemPerMultiprocessor | 102,400 B | device_info |
| regsPerSM | 65,536 | device_info |
| maxThreadsPerSM | 1,536 | device_info |
| clusterLaunch | 1 (clusters supported) | device_info |
| driver / toolkit | 595.99.02 / CUDA 13.3.1 | nvidia-smi / nvcc |
| CUTLASS | v4.3.0, tag commit e67e63c3 | version.h + GitHub tag |
| ncu | 2026.2.1 present, **counters denied** | ERR_NVGPUCTRPERM |
| nsys | 13.3.1 present (works without counters) | - |

## Native FP4 evidence: PASSED (SASS)
`cuobjdump -dump-sass` on 79a yields **64 instances** of
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`
= m16n8k64 block-scaled MMA, E2M1 operands, UE4M3 block scale, .SF modifier.
This is the SM120 native block-scaled path, not dequant-to-BF16 emulation.

## 79a execution: BLOCKED (measured)
```
CUTLASS_DEBUG_TRACE_LEVEL=1 ->
  Setting smem size to 88064
  cudaFuncSetAttribute() returned error: invalid resource handle
  Got cutlass error: Error Internal at: 450
```
- 88064 B < 101376 B, so it is **not** a shared-memory overflow.
- Minimal control: `cudaFuncSetAttribute(k, MaxDynamicSharedMemorySize, 88064)`
  returns success for a trivial kernel on the same driver/arch, and launching
  with 88 KiB dynamic smem + 170 blocks works.
- Reproduced identically with hand-written nvcc and official CMake builds.

Known upstream evidence that NVFP4-on-SM120 is immature:
- CUTLASS issue #2906: 79a crashes (misaligned address) on RTX 5090.
- vLLM PR #29711: `fp4 scaled_mm` fails on 5090 with "Internal Error".
- vLLM PR #48965: don't auto-select CUTLASS NVFP4 kernels on SM 12x.
- DGX-Spark report: SM120 GEMM tiles designed for 228 KiB SMEM overflow the 99 KiB limit.

=> M0 item "79a runs with Disposition: Passed" = **CONDITIONAL / NOT MET**.

## Quantization recipe: PASSED (reference implementation)
`src/nvfp4_ref.py` + `tests/test_nvfp4_ref.py` (16/16 PASS):
- E2M1 set {0,+-0.5,+-1,+-1.5,+-2,+-3,+-4,+-6}, saturation verified
- E4M3 block scale on the real e4m3fn grid, max 448
- round-trip relative L2: normal 9.6%, long-tail 8.1%, uniform 10.2%
- tiny GEMM double gold standard: FP4-vs-BF16 relative L2 = 14.6%
Not yet done: CUTLASS SFA/SFB swizzled layout (needed before G3 can be compared).

## G0 baseline harness: WORKING (first numbers, shared GPU - not final)
B = 12288x4096 BF16 = 96 MiB (== L2), K=4096, cuBLASLt, 20 samples x 10 launches:

| M | p50 us | eff TFLOPS | note |
|---|---|---|---|
| 16 | 65.2 | 24.7 | bandwidth-bound: 96 MiB / 65 us = 1.55 TB/s (~86% of 1792 GB/s) |
| 128 | 65.5 | 196.9 | same time as M=16 => still bandwidth-bound |
| 512 | 257.1 | 200.5 | 4x work / 4x time => compute-bound at ~48% of BF16 peak |

Crossover between M=128 and M=512 - a first, usable E1 observation for BF16.
Numbers are taken while another user's job holds ~24 GB; not final.

## Blockers to open
1. NCU counters: need `NOPASSWD: /usr/local/cuda/bin/ncu *` (do not reload the driver).
2. Exclusive GPU window for all formal timing.
3. Decide G3 fallback: another CUTLASS version / marlin NVFP4 / custom kernel.

---

## Addendum: G0 BF16 sweep on a FREE GPU (17:2x, another user's job had ended)

GPU state before/after: 347 MiB used, only Xorg + gnome-shell, 23 W idle -> 120 W, 54->60 C.
**This is the first measuring window with no competing compute process.**

### Main shape, K=4096 N=12288 (B = 96 MiB = exactly L2 size)

| M | p50 us | p90 us | eff TFLOPS | B bytes / p50 = implied BW |
|---:|---:|---:|---:|---:|
| 16 | 65.2 | 65.3 | 24.7 | 1.54 TB/s |
| 32 | **43.9** | 44.1 | 73.3 | **2.30 TB/s (impossible from DRAM -> L2 hits)** |
| 64 | **80.8** | 81.1 | 79.8 | 1.24 TB/s |
| 128 | 65.5 | 65.6 | 196.7 | 1.53 TB/s |
| 256 | 120.9 | 121.2 | 213.1 | 0.83 TB/s |
| 512 | 257.8 | 258.7 | 199.9 | 0.39 TB/s |
| 1024 | 474.7 | 475.0 | 217.1 | 0.21 TB/s |

**Finding 1 (methodological red flag):** time is NOT monotonic in M — M=32 (43.9 us) is
faster than M=16 (65.2 us) and M=64 (80.8 us). cuBLASLt picks different tile kernels per M,
and 2.30 TB/s exceeds the 1.79 TB/s DRAM peak, i.e. L2 reuse is already happening at M=32.
=> any BF16-vs-FP4 comparison at small M must first fix/record the kernel choice, otherwise
the measured "precision difference" is partly a kernel-selection difference (exactly the
confound V2 §0/§3.5 warns about).

### L2 breakpoint, K=4096, M=128/512

| M | N | B bytes | p50 us | eff TFLOPS |
|---:|---:|---:|---:|---:|
| 128 | 4096 | 32 MiB | 28.7 | 149.6 |
| 128 | 16384 | 128 MiB | 95.8 | 179.2 |
| 128 | 65536 | 512 MiB | 475.3 | 144.6 |
| 512 | 4096 | 32 MiB | 82.4 | 208.4 |
| 512 | 16384 | 128 MiB | 326.1 | 210.8 |
| 512 | 65536 | 512 MiB | 1835.1 | 149.8 |

BF16 peak here is ~217 TFLOPS (52% of the 419 TFLOPS dense spec) - cuBLASLt never exceeds
this on this shape set, so "50% of peak" is the practical BF16 ceiling to beat.

---

## Addendum 2: G3 route attempts (all fired today)

### CUTLASS version sweep
| version | build | run |
|---|---|---|
| v4.3.0 (nvcc ad-hoc) | OK, 57 s | `Error Internal` at `gemm.initialize` |
| v4.3.0 (official CMake) | OK | same |
| v4.2.0 (nvcc ad-hoc) | OK, 54 s | **same** -> not version-specific |

Root cause refined: `cudaFuncSetAttribute(device_kernel<GemmKernel>, 88064)` returns
`invalid resource handle` while a trivial kernel with the same 88 KiB dynamic SMEM
succeeds on the same driver/arch. Kernel symbol in the binary has *internal linkage*
(`_ZN7cutlassL...13device_kernel...`, lowercase `t`).
=> 79a is not a viable G3 baseline on this toolchain; stop investing here.

### torch 2.15.0+cu132 (installed from TUNA mirror)
- `torch.cuda.is_available() == True`, device cc (12,0), BF16 matmul verified on GPU.
- `torch.float4_e2m1fn_x2` and `torch._scaled_mm` both exist.
- NVFP4 1x16 path (`a,b` = float4 packed, scales = float8_e4m3fn, 2048 elements):
  the kernel **executes** on sm_120 (no error, correct output shape), but the result
  does not yet match a reference built from the identical payload/scale bits
  (relative L2 ~0.56). Most likely a packing/layout convention mismatch on our side
  (which nibble is which element, and whether mat2 is (K,N/2)-packed or (N,K)-packed).
  => status: **runs, numerics unverified**, NOT accepted as a G3 baseline yet.

### Conclusion for G3
Order of preference now: (1) finish the torch NVFP4 packing convention (cheapest, one
evening), (2) marlin-style NVFP4 kernel, (3) hand-written `mma.sync` PTX kernel.

---

## Addendum 3: G3 UNBLOCKED - torch native NVFP4 GEMM verified (17:5x)

Root cause of the earlier 0.56 relative error: **scale-factor layout**, not dtype/count.
`_meta_registrations.py:7374-7387` requires the L4-padded SWIZZLE_32_4_4 layout:
`padded_k_blocks = ceil(kb/4)*4`, `numel = 128*ceil(M/128)*padded_k_blocks`.
At our shapes the element COUNT coincides with plain row-major (128*16 = 2048), so the
size check passed while the arrangement was wrong - a trap that also holds for K=4096.

Fix: permute the scales exactly as `flex_gemm::to_blocked` does
(pad -> reshape(rb,128,cb,4).permute(0,2,1,3) -> reshape(-1,4,32,4).transpose(1,2) -> flatten),
pass both as `.reshape(-1)`, use `weight.t()`, then multiply the global alpha.

Staged probe (tests/nvfp4_layout_probe.py), all stages PASS:
| stage | test | result |
|---|---|---|
| 1 | all FP4=1, all scales=1, K=256 | out = **exactly 256.0** |
| 2 | one-hot at k=0,1,16,17 | out[0,0] = exactly 1.0 (packing order + K order correct) |
| 3 | block scale at kb=0 and kb=5 | out[0,0] = exactly 2.0 (swizzle indexing correct) |
| 4 | row-dependent scale (row1 x4) | PASS |
| 5 | random payload + random scales | rel_L2 = 1.67e-3 (bf16 output rounding) |

Also confirmed from source: packing is `(code[k+1]<<4) | code[k]` (even k -> low nibble).

Note: this was run while another user held 24 GB (no system restriction on sharing a
consumer GPU); it used a few MB and took 1.9 s wall, so it does not disturb timing work.
Formal timing still requires an unloaded GPU.
