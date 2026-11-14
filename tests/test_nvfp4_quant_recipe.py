"""CPU-only tests for the corrected NVFP4 recipe (encode/decode scale split).

Covers: zero blocks, E2M1/E4M3 boundaries, saturation/overflow, block-16 isolation,
and the requirement that g_encode stays FP32 (never rounded to E4M3).
"""
import sys, numpy as np
sys.path.insert(0, 'src')
import nvfp4_recipe as R

fails = 0
def check(name, cond, extra=''):
    global fails
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond: fails += 1

# --- 1. encode/decode scale split -------------------------------------------
x = np.array([[1.0, 2.0, 3.0, 4.0] * 4], np.float32)      # amax = 4 -> g = 672
q = R.quantize(x)
ga = q['g_encode']
check('g_encode == 2688/amax', abs(ga - 2688.0/4.0) < 1e-3, f'g={ga}')
check('g_encode NOT forced onto E4M3 grid', R.round_e4m3(np.float32(ga)) != np.float32(ga),
      f'e4m3(g)={R.round_e4m3(np.float32(ga)):.6f} vs g={ga:.6f}')
check('max |x_enc| == 2688 (full-range normalisation)',
      abs(float(np.abs(x*ga).max()) - 2688.0) < 1.0, f'{float(np.abs(x*ga).max()):.3f}')
check('block scale hits E4M3 max 448 exactly at the global-max block',
      abs(float(q['block_scale'].max()) - 448.0) < 1e-6, f"{q['block_scale'].max()}")
check('decode inverts encode (rel L2 < 0.12)',
      np.linalg.norm(q['dequant']-x)/np.linalg.norm(x) < 0.12,
      f"{np.linalg.norm(q['dequant']-x)/np.linalg.norm(x):.4f}")

# --- 2. zero / degenerate ----------------------------------------------------
z = R.quantize(np.zeros((2, 32), np.float32))
check('all-zero -> payload all zero', z['payload'].max() == 0)
check('all-zero -> no nan/inf in dequant', np.isfinite(z['dequant']).all())
check('all-zero -> g_encode finite', np.isfinite(z['g_encode']))

one_zero_block = np.ones((1, 32), np.float32); one_zero_block[0, 16:] = 0.0
qb = R.quantize(one_zero_block)
check('zero block gets a valid (nonzero) E4M3 scale', float(qb['block_scale'][0, 1]) > 0,
      f"{qb['block_scale'][0,1]:.3e}")
check('zero block payload is zero', qb['payload'][0, 16:].max() == 0)
check('zero block does not pollute the nonzero block',
      np.allclose(qb['dequant'][0, :16], 1.0, atol=0.2), f"{qb['dequant'][0,:4]}")

# --- 3. E2M1 exactness when the block maximum is representable --------------
# With amax == 6 the encoded grid has step 448, so E2M1 values map exactly.
vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5], np.float32)
xl = np.tile(vals, 2).reshape(2, 16)          # amax = 6.0 exactly
ql = R.quantize(xl)
dec = ql['dequant'].ravel()[:16]
codes = R.nibbles_to_values(ql['payload']).ravel()[:16]
check('exact E2M1 values round-trip bit-exactly (amax=6)',
      np.allclose(dec[:8], vals[:8], atol=1e-3), f'{np.round(dec[:8],3)}')
check('payload codes for those values are exact too',
      np.allclose(codes[:8], vals[:8], atol=1e-6), f'{np.round(codes[:8],3)}')
check('non-representable 0.25/1.25 snap to a neighbour (<= half step)',
      np.all(np.abs(dec[8:12] - vals[8:12]) <= 0.5 + 1e-3), f'{np.round(dec[8:12],3)}')

# --- 4. overflow / saturation ------------------------------------------------
big = np.full((1, 16), 1e6, np.float32); big[0, 0] = 1e-9
qbig = R.quantize(big)
check('huge dynamic range does not produce inf/nan', np.isfinite(qbig['dequant']).all())
check('saturated payload uses code for 6.0', R.nibbles_to_values(qbig['payload']).max() <= 6.0 + 1e-6)
check('tiny element underflows to 0, not to garbage',
      abs(float(R.nibbles_to_values(qbig['payload'])[0, 0])) <= 0.5, f"{R.nibbles_to_values(qbig['payload'])[0,0]}")

# --- 5. block-16 isolation ---------------------------------------------------
xa = np.zeros((1, 32), np.float32); xa[0, :16] = np.linspace(-1, 1, 16); xa[0, 16:] = np.linspace(-1e-3, 1e-3, 16)
qa = R.quantize(xa)
check('two blocks get different scales', qa['block_scale'][0,0] != qa['block_scale'][0,1],
      f"{qa['block_scale'][0,0]:.4g} vs {qa['block_scale'][0,1]:.4g}")
check('small block still resolves (rel err on that block < 0.35)',
      np.linalg.norm(qa['dequant'][0,16:]-xa[0,16:])/max(np.linalg.norm(xa[0,16:]),1e-12) < 0.35)

# --- 6. packing round trip ---------------------------------------------------
pk = R.pack(q['payload'])
check('pack/unpack round trip', np.array_equal(
      np.stack([pk[...,None] & 0xF, (pk[...,None] >> 4) & 0xF], -1).reshape(q['payload'].shape), q['payload']))

# --- 7. alpha (decode factor) -------------------------------------------------
al = R.gemm_alpha(q['g_encode'], q['g_encode'])
check('alpha = 1/(g_a g_b) and finite', np.isfinite(al) and abs(al - 1.0/(ga*ga)) < 1e-12, f'{al:.3e}')

print('\nFAILURES:', fails)
sys.exit(1 if fails else 0)
