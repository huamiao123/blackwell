"""NVFP4 (1x16 block, e4m3 scales, e2m1 payload) GEMM through torch._scaled_mm.
Compares against a numpy reference built from the exact same payload+scale bits."""
import numpy as np, torch, sys
E2M1 = np.array([0,0.5,1,1.5,2,3,4,6], dtype=np.float32)
def decode(u8):
    lo = u8 & 0xF; hi = (u8 >> 4) & 0xF
    nib = np.stack([lo,hi],axis=-1).reshape(*u8.shape[:-1],-1)
    return (E2M1[np.minimum(nib & 7,7)] * np.where(nib & 8,-1.0,1.0)).astype(np.float32)

M,K,N = 128,256,128
rng=np.random.default_rng(0)
a_u8=rng.integers(0,256,(M,K//2),dtype=np.uint8)
b_u8=rng.integers(0,256,(N,K//2),dtype=np.uint8)          # B^T layout (N,K)
sa=(rng.uniform(.02,.08,(M,K//16))).astype(np.float32)
sb=(rng.uniform(.02,.08,(N,K//16))).astype(np.float32)
A=decode(a_u8)*np.repeat(sa,16,axis=1)
B=decode(b_u8)*np.repeat(sb,16,axis=1)
ref=A@B.T

ta=torch.from_numpy(a_u8).cuda().view(torch.float4_e2m1fn_x2)
tb=torch.from_numpy(b_u8).cuda().view(torch.float4_e2m1fn_x2)
tsa=torch.from_numpy(sa).cuda().to(torch.float8_e4m3fn)
tsb=torch.from_numpy(sb).cuda().to(torch.float8_e4m3fn)
try:
    out=torch._scaled_mm(ta,tb.T,scale_a=tsa,scale_b=tsb,out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    o=out.float().cpu().numpy()
    rel=np.linalg.norm(o-ref)/(np.linalg.norm(ref)+1e-9)
    print("SUCCESS: NVFP4 1x16 GEMM ran on sm_120 via torch._scaled_mm")
    print(f"  shape={tuple(o.shape)} relative_L2_vs_reference={rel:.4e} ref_absmax={np.abs(ref).max():.3f}")
    print("VERDICT:", "PASS" if rel < 5e-2 else f"SUSPICIOUS rel={rel}")
    sys.exit(0 if rel < 5e-2 else 1)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:600]); sys.exit(2)
