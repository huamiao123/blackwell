"""CPU-only: prove the stage-6 comparison is self-consistent.

stage-6 compares the kernel output against
    ref_mm(payload, scale, alpha) = (q*sl) @ (q*sl).T * alpha
The independent reference is
    dequant(payload, scale, g) @ dequant(payload, scale, g).T   with dequant = q*sl/g
They must agree exactly when alpha = 1/(g_a*g_b). If they disagree, the acceptance
threshold would be measuring our own bookkeeping error instead of the kernel.
"""
import sys, numpy as np
sys.path.insert(0, 'src')
import nvfp4_recipe as R

def ref_mm(a_nib, w_nib, sa, sb, alpha):
    A = R.nibbles_to_values(a_nib) * np.repeat(sa, 16, axis=1)
    W = R.nibbles_to_values(w_nib) * np.repeat(sb, 16, axis=1)
    return (A @ W.T) * alpha

fails = 0
def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond: fails += 1

rng = np.random.default_rng(11)
for shape in [(128, 256, 128), (64, 128, 64), (16, 64, 32)]:
    M, K, N = shape
    A = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.standard_normal((N, K)).astype(np.float32)
    qa, qb = R.quantize(A), R.quantize(W)
    alpha = R.gemm_alpha(qa['g_encode'], qb['g_encode'])
    r1 = ref_mm(qa['payload'], qb['payload'], qa['block_scale'], qb['block_scale'], alpha)
    r2 = qa['dequant'] @ qb['dequant'].T
    err = float(np.abs(r1 - r2).max())
    rel = float(np.linalg.norm(r1 - r2) / max(np.linalg.norm(r2), 1e-30))
    check(f"reference self-consistency {shape}", rel < 1e-6, f"max_abs={err:.3e} rel={rel:.3e}")

# zeros must stay zero through the whole path
Z = np.zeros((32, 64), np.float32)
qz = R.quantize(Z)
a = R.gemm_alpha(qz['g_encode'], qz['g_encode'])
r = ref_mm(qz['payload'], qz['payload'], qz['block_scale'], qz['block_scale'], a)
check("all-zero matrices give exactly zero", np.all(r == 0))

# a single hot element must land at the right coordinate
A = np.zeros((32, 64), np.float32); A[3, 17] = 6.0
W = np.zeros((32, 64), np.float32); W[5, 17] = 6.0
qa, qb = R.quantize(A), R.quantize(W)
al = R.gemm_alpha(qa['g_encode'], qb['g_encode'])
r = ref_mm(qa['payload'], qb['payload'], qa['block_scale'], qb['block_scale'], al)
hot = np.argwhere(np.abs(r) > 1e-6).tolist()
check("single hot element maps to (3,5)", hot == [[3, 5]], f"{hot} value={r[3,5]:.4f}")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
