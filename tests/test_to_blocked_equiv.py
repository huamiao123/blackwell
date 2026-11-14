"""CPU-only: verify our numpy SWIZZLE_32_4_4 against torch's own to_blocked.

No GPU is touched. Shapes deliberately include padding cases:
  rows not a multiple of 128, cols not a multiple of 4, tiny shapes.
"""
import numpy as np, torch, sys
import torch._higher_order_ops.flex_gemm  # registers flex_gemm::to_blocked
from torch._higher_order_ops.flex_gemm import to_blocked as torch_to_blocked

def to_blocked_np(s2d):
    rows, cols = s2d.shape
    rb, cb = (rows+127)//128, (cols+3)//4
    pad = np.zeros((rb*128, cb*4), np.float32); pad[:rows, :cols] = s2d
    b = pad.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3)
    return b.reshape(-1)

fails = 0
rng = np.random.default_rng(0)
shapes = [(128,16),(128,256),(128,1),(1,1),(37,5),(129,7),(256,3),(100,255),(383,257)]
print(f"{'shape':>12} {'numpy_n':>9} {'torch_n':>9} {'max_abs_diff':>14} {'status':>7}")
for rows, cols in shapes:
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    ref = to_blocked_np(x)
    t = torch.tensor(x)
    got = torch_to_blocked(t).numpy().astype(np.float32)
    same_len = ref.size == got.size
    diff = float(np.abs(ref - got).max()) if same_len else float('nan')
    ok = same_len and diff == 0.0
    print(f"{str((rows,cols)):>12} {ref.size:9d} {got.size:9d} {diff:14.3e} {'PASS' if ok else 'FAIL':>7}")
    if not ok: fails += 1

# also verify the element mapping explicitly on a 128x8 input: value == index
x = np.arange(128*8, dtype=np.float32).reshape(128, 8)
ref = to_blocked_np(x); got = torch_to_blocked(torch.tensor(x)).numpy().astype(np.float32)
ok = ref.size == got.size and np.array_equal(ref, got)
print(f"{'index-map':>12} {ref.size:9d} {got.size:9d} {0 if ok else 1:14.3e} {'PASS' if ok else 'FAIL':>7}")
if not ok: fails += 1

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
