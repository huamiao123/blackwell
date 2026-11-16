"""BF16 MMA compute probe. Fixed shape; inner MMA repetition is explicit work."""
import triton
import triton.language as tl


@triton.jit
def _aux_tensor(A, B, C, Stamp, inner, CHUNK,
                M: tl.constexpr = 64, N: tl.constexpr = 128, K: tl.constexpr = 1024):
    rm = tl.program_id(0) * 16 + tl.arange(0, 16)
    rn = tl.program_id(1) * 32 + tl.arange(0, 32)
    rk = tl.arange(0, 32)
    acc = tl.full((16, 32), 0, tl.float32)
    for k in range(K // 32):
        av = tl.load(A + rm[:, None] * K + k * 32 + rk[None, :])
        bv = tl.load(B + (k * 32 + rk[:, None]) * N + rn[None, :])
        for _ in range(inner):
            acc = tl.dot(av, bv, acc)
    tl.store(C + CHUNK * M * N + rm[:, None] * N + rn[None, :], acc)
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.store(Stamp + CHUNK, CHUNK + 1)


def launch(a, b, c, stamp, inner, chunk):
    return _aux_tensor[(4, 4)](a, b, c, stamp, inner, chunk, num_warps=4, num_stages=1)
