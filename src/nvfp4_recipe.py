"""NVFP4 1D block-16 recipe (vectorised) with explicit encode/decode scale split.

encode:  g = 2688 / amax        (FP32, never rounded to E4M3)
         x_enc = x * g
         s_block = amax(|x_enc| in 16) / 6      -> FP8 E4M3
         q = round_e2m1(x_enc / s_block)        -> E2M1, saturating
decode:  x_hat = q * s_block / g
GEMM:    y = (q_a s_a).(q_b s_b) * 1/(g_a g_b)
"""
import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
BLOCK = 16
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0 ** -9
QMAX = 6.0
ENCODE_NORM = E4M3_MAX * QMAX

def _e4m3_table():
    vals = []
    for e in range(16):
        for m in range(8):
            bits = (e << 3) | m
            if bits == 0x7F:
                continue
            vals.append(m / 8.0 * 2.0 ** (1 - 7) if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.array(sorted(vals), np.float64)
_E4M3 = _e4m3_table()

def _snap_mag(a, table, cap):
    idx = np.clip(np.searchsorted(table, a), 1, len(table) - 1)
    lo, hi = table[idx - 1], table[idx]
    return np.where(a >= cap, cap, np.where(a - lo <= hi - a, lo, hi))

def round_e4m3(x):
    x = np.asarray(x, np.float64)
    m = _snap_mag(np.abs(x), _E4M3, E4M3_MAX)
    return np.where(x < 0, -m, m)

def round_e2m1(x):
    x = np.asarray(x, np.float64)
    m = _snap_mag(np.abs(x), E2M1, QMAX)
    return np.where(x < 0, -m, m)

def _codes_from_mag(m):
    idx = np.clip(np.searchsorted(E2M1, m), 1, len(E2M1) - 1)
    lo, hi = E2M1[idx - 1], E2M1[idx]
    return np.where(m - lo <= hi - m, idx - 1, idx).astype(np.uint8)

def quantize(x):
    x = np.asarray(x, np.float32)
    rows, cols = x.shape
    if cols % BLOCK:
        raise ValueError(f"cols={cols} must be a multiple of {BLOCK}")
    amax = float(np.abs(x).max())
    nb = cols // BLOCK
    if amax == 0.0:
        return dict(payload=np.zeros((rows, cols), np.uint8),
                    block_scale=np.full((rows, nb), E4M3_MIN_SUBNORMAL, np.float32),
                    g_encode=1.0, amax=0.0, dequant=np.zeros_like(x))
    g = np.float32(ENCODE_NORM / amax)
    x_enc = (x * g).astype(np.float64)
    blk = x_enc.reshape(rows, nb, BLOCK)
    bmax = np.abs(blk).max(axis=-1)
    s = _snap_mag(bmax / QMAX, _E4M3, E4M3_MAX)
    s = np.where(s <= 0, E4M3_MIN_SUBNORMAL, s)
    q = blk / s[:, :, None]
    mag = np.minimum(np.abs(q), QMAX)
    codes = _codes_from_mag(mag)
    payload = (codes | np.where(q < 0, 0x8, 0)).astype(np.uint8).reshape(rows, cols)
    scale = s.astype(np.float32)
    return dict(payload=payload, block_scale=scale, g_encode=float(g), amax=amax,
                dequant=dequantize(payload, scale, g))

def nibbles_to_values(nib):
    nib = np.asarray(nib, np.uint8)
    return (E2M1[nib & 7] * np.where(nib & 0x8, -1.0, 1.0)).astype(np.float32)

def dequantize(payload, block_scale, g_encode):
    v = nibbles_to_values(payload) * np.repeat(np.asarray(block_scale, np.float32), BLOCK, axis=1)
    return (v / np.float32(g_encode)).astype(np.float32)

def pack(payload):
    assert payload.shape[-1] % 2 == 0
    return ((payload[..., 1::2] << 4) | payload[..., 0::2]).astype(np.uint8)

def gemm_alpha(g_a, g_b):
    return float(np.float32(1.0) / (np.float32(g_a) * np.float32(g_b)))