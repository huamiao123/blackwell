import random, math, sys
sys.path.insert(0, 'src')
from nvfp4_ref import *

fails = 0
def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + ('  ' + extra if extra else ''))
    if not cond: fails += 1

# 1) E2M1 rounding lands on the representable set
check('e2m1 table size', len(E2M1_VALUES) == 15, str(E2M1_VALUES))
vals = [round_e2m1(random.uniform(-8, 8)) for _ in range(2000)]
check('e2m1 outputs representable', all(v in E2M1_VALUES for v in vals))
check('e2m1 saturates', round_e2m1(100.0) == 6.0 and round_e2m1(-100.0) == -6.0)

# 2) E4M3 rounding sanity
check('e4m3 max clamp', abs(round_e4m3(1e6)) <= E4M3_MAX)
check('e4m3 exact on grid', round_e4m3(448.0) == 448.0 and round_e4m3(0.5) == 0.5)

# 3) quantize/dequantize error bounds (normal + long tail)
random.seed(42)
for dist, gen in [('normal', lambda: random.gauss(0, 1)),
                  ('longtail', lambda: random.gauss(0, 1) ** 3),
                  ('uniform', lambda: random.uniform(-1, 1))]:
    x = [gen() for _ in range(4096)]
    q = quantize_nvfp4(x)
    m = metrics(x, q['deq'])
    # NVFP4 has 4-bit payload: relative L2 should be small but not tiny
    check(f'roundtrip {dist} relative_L2 < 0.15', m['relative_L2'] < 0.15, f"{m['relative_L2']:.4f}")
    check(f'roundtrip {dist} no nan/inf', m['nan_inf'] == 0)
    check(f'roundtrip {dist} payload 4-bit', all(v in E2M1_VALUES for v in q['payload']))

# 4) double gold standard on a tiny GEMM (M=8,K=64,N=64)
M, K, N = 8, 64, 64
random.seed(7)
A = [to_bf16(random.gauss(0, 1)) for _ in range(M*K)]
B = [to_bf16(random.gauss(0, 1)) for _ in range(K*N)]
ref_original = gemm_fp32(A, B, M, N, K)
qa, qb = quantize_nvfp4(A), quantize_nvfp4(B)
ref_quant = gemm_fp32(qa['deq'], qb['deq'], M, N, K)
m_q = metrics(ref_original, ref_quant)
check('fp4-vs-bf16 gemm quant error < 40% rel L2', m_q['relative_L2'] < 0.40, f"{m_q['relative_L2']:.4f}")

# 5) zero tensor degenerate case
q0 = quantize_nvfp4([0.0]*64)
check('all-zero tensor -> all-zero payload', all(v == 0.0 for v in q0['deq']))

print()
print('FAILURES:', fails)
sys.exit(1 if fails else 0)
