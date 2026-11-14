"""Staged probe: make torch's native NVFP4 GEMM agree with a decoded reference.

Key facts extracted from the installed torch source (2.15.0+cu132):
  * packing (torch/_higher_order_ops/flex_gemm.py:184): packed = (code[...,1]<<4) | code[...,0]
    => element 2p goes in the LOW nibble, element 2p+1 in the HIGH nibble.
  * scales must be the enums' layout, NOT row-major:
      torch/_meta_registrations.py:7374-7387 ->
        padded_num_k_blocks = ceil(k_blocks/4)*4
        expected numel = 128*ceil(M/128)*padded_k_blocks   (same for N)
      torch/_higher_order_ops/flex_gemm.py:188-216 `to_blocked`:
        pad to (rb*128, cb*4); reshape(rb,128,cb,4).permute(0,2,1,3)
        .reshape(-1,4,32,4).transpose(1,2).reshape(-1)
  * vLLM usage: weight.t(), both scales .reshape(-1), then out * alpha.
Stages (stop at first failure):
  1 all-ones payload + all-ones scales      -> expect exactly K
  2 single one-hot element                  -> packing order / transpose
  3 per-block scales at kb=0 and kb=5       -> swizzle indexing
  4 row-dependent scales                    -> row mapping
  5 random payload + random scales vs numpy -> end-to-end
"""
import numpy as np, torch, sys
import torch._higher_order_ops.flex_gemm  # registers flex_gemm::to_blocked

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
CODE = {v: i for i, v in enumerate(E2M1)}
def f32_to_nib(x):  return np.vectorize(lambda v: CODE[float(v)])(np.asarray(x, np.float32)).astype(np.uint8)
def nib_to_f32(n):  return E2M1[n & 7] * np.where(n & 8, -1.0, 1.0)
def pack(nib):                                  # nib[...,K] -> uint8[...,K/2], low nibble = even k
    assert nib.shape[-1] % 2 == 0
    return ((nib[..., 1::2] << 4) | nib[..., 0::2]).astype(np.uint8)
def unpack(u8):
    lo = (u8 & 0x0F).astype(np.uint8); hi = ((u8 >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(u8.shape[:-1] + (u8.shape[-1]*2,), np.uint8)
    out[..., 0::2] = lo; out[..., 1::2] = hi
    return out

def to_blocked_np(s2d):                          # numpy twin of flex_gemm.to_blocked
    rows, cols = s2d.shape
    rb, cb = (rows+127)//128, (cols+3)//4
    pad = np.zeros((rb*128, cb*4), np.float32); pad[:rows, :cols] = s2d
    b = pad.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3)
    return b.reshape(-1)

def f32_to_e4m3_bytes(x):
    t = torch.from_numpy(np.asarray(x, np.float32)).to(torch.float8_e4m3fn)
    return t.view(torch.uint8).numpy(), t.float().numpy()

def run_case(M, K, N, a_nib, b_nib, sa, sb, label, expect=None):
    """a_nib (M,K) nibbles, b_nib (N,K) nibbles (weight layout), sa (M,K/16), sb (N,K/16) fp32."""
    dev = "cuda"
    ta = torch.from_numpy(pack(a_nib)).to(dev).view(torch.float4_e2m1fn_x2)
    tw = torch.from_numpy(pack(b_nib)).to(dev).view(torch.float4_e2m1fn_x2)
    sab, sa_r = f32_to_e4m3_bytes(sa); sbb, sb_r = f32_to_e4m3_bytes(sb)
    tsa = torch.from_numpy(to_blocked_np(sa).astype(np.float32)).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    tsb = torch.from_numpy(to_blocked_np(sb).astype(np.float32)).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    out = torch._scaled_mm(ta, tw.t(), scale_a=tsa, scale_b=tsb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    o = out.float().cpu().numpy()
    A = nib_to_f32(a_nib) * np.repeat(sa_r, 16, axis=1)
    B = nib_to_f32(b_nib) * np.repeat(sb_r, 16, axis=1)
    ref = A @ B.T
    rel = np.linalg.norm(o-ref)/(np.linalg.norm(ref)+1e-9)
    ok = rel < 2e-2
    print(f"[{label}] rel_L2={rel:.3e} out[0,0]={o[0,0]:.4f} ref[0,0]={ref[0,0]:.4f} {'PASS' if ok else 'FAIL'}")
    if expect is not None:
        print(f"        expect {expect} got {o[0,0]:.4f} {'OK' if abs(o[0,0]-expect)<1e-2 else 'MISMATCH'}")
    return ok

M, K, N = 128, 256, 128
kb = K//16
ones = np.full((M, K), CODE[1.0], np.uint8)
sa1 = np.ones((M, kb), np.float32); sb1 = np.ones((N, kb), np.float32)

print("=== stage 1: all ones, all scales 1 ===")
ok1 = run_case(M, K, N, ones, ones, sa1, sb1, "ones", expect=float(K))
if not ok1:
    print("STOP: even the trivial all-ones case is wrong -> packing/transpose/layout issue")
    sys.exit(1)

print("=== stage 2: one-hot at k=0 and k=1 ===")
for pos in (0, 1, 16, 17):
    a = np.zeros((M, K), np.uint8); a[0, pos] = CODE[1.0]
    b = np.zeros((N, K), np.uint8); b[0, pos] = CODE[1.0]
    run_case(M, K, N, a, b, sa1, sb1, f"onehot_k{pos}", expect=1.0)

print("=== stage 3: scale only at kb=0 then kb=5 (a side) ===")
for kbi in (0, 5):
    sa = np.ones((M, kb), np.float32); sa[0, :] = 1.0; sa[0, kbi] = 2.0
    a = np.zeros((M, K), np.uint8); a[0, 16*kbi] = CODE[1.0]
    b = np.zeros((N, K), np.uint8); b[0, 16*kbi] = CODE[1.0]
    run_case(M, K, N, a, b, sa, sb1, f"scale_kb{kbi}", expect=2.0)

print("=== stage 4: row-dependent scale ===")
sa = np.ones((M, kb), np.float32); sa[1, :] = 4.0
a = np.zeros((M, K), np.uint8); a[0, 0] = CODE[1.0]; a[1, 0] = CODE[1.0]
b = np.zeros((N, K), np.uint8); b[0, 0] = CODE[1.0]
ok4 = run_case(M, K, N, a, b, sa, sb1, "row_scale", expect=1.0)

print("=== stage 5: random payload + random scales ===")
rng = np.random.default_rng(0)
a_nib = rng.integers(0, 16, (M, K)).astype(np.uint8)
b_nib = rng.integers(0, 16, (N, K)).astype(np.uint8)
sa = rng.uniform(.01, .1, (M, kb)).astype(np.float32)
sb = rng.uniform(.01, .1, (N, kb)).astype(np.float32)
ok5 = run_case(M, K, N, a_nib, b_nib, sa, sb, "random")
print()
print("RESULT:", "ALL_PASS" if all([ok1, ok4, ok5]) else "PARTIAL")
sys.exit(0 if all([ok1, ok4, ok5]) else 1)
