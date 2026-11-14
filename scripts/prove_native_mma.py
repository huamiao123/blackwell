"""Which kernel actually runs for torch._scaled_mm on float4_e2m1fn_x2?
Prints the CUDA kernel names from torch.profiler and writes them to a log.
A native FP4 kernel name (cutlass/nvjet/...fp4/blocked/scale) is the first half
of the proof; SASS confirmation is done by scripts/prove_native_mma.sh.
GPU REQUIRED - run only in an approved window.
"""
import sys, json
import numpy as np, torch
try:
    import torch._higher_order_ops.flex_gemm  # not present in older torch; unused here
except Exception:
    pass

def to_blocked_np(s2d):
    rows, cols = s2d.shape
    rb, cb = (rows+127)//128, (cols+3)//4
    pad = np.zeros((rb*128, cb*4), np.float32); pad[:rows, :cols] = s2d
    return pad.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3).reshape(-1)

M,K,N = 256,512,256
rng = np.random.default_rng(0)
a = torch.from_numpy(rng.integers(0,256,(M,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
w = torch.from_numpy(rng.integers(0,256,(N,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
sa = torch.from_numpy(to_blocked_np(rng.uniform(.01,.1,(M,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
sb = torch.from_numpy(to_blocked_np(rng.uniform(.01,.1,(N,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        torch._scaled_mm(a, w.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()

names = []
for evt in prof.key_averages():
    if evt.device_time_total > 0:
        names.append((evt.key, evt.count, evt.device_time_total))
names.sort(key=lambda x: -x[2])
print(f"{'kernel':<90} {'count':>6} {'cuda_us':>10}")
for k, c, t in names[:20]:
    print(f"{k[:90]:<90} {c:6d} {t:10.1f}")
json.dump([{'kernel': k, 'count': c, 'cuda_us': t} for k, c, t in names],
          open('raw/native_mma_kernels.json', 'w'), indent=2)
print('\njson -> raw/native_mma_kernels.json')
