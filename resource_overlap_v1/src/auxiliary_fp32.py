"""Low-traffic, real FP32 FMA probe; runtime loop cannot be constant folded."""
import triton
import triton.language as tl


@triton.jit
def _aux_fp32(X, Y, Stamp, reps, chunk, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i)
    a, b, c, d = x, x + 0.125, x + 0.25, x + 0.375
    for _ in range(reps):
        a = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, 0f3f800001, 0f33d6bf95;", constraints="=f,f", args=[a], dtype=tl.float32, is_pure=False, pack=1)
        b = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, 0f3f800002, 0f33d6bf95;", constraints="=f,f", args=[b], dtype=tl.float32, is_pure=False, pack=1)
        c = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, 0f3f7ffffd, 0f33d6bf95;", constraints="=f,f", args=[c], dtype=tl.float32, is_pure=False, pack=1)
        d = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, 0f3f7ffff9, 0f33d6bf95;", constraints="=f,f", args=[d], dtype=tl.float32, is_pure=False, pack=1)
    base = chunk * tl.num_programs(0) * BLOCK * 4 + i * 4
    tl.store(Y + base, a)
    tl.store(Y + base + 1, b)
    tl.store(Y + base + 2, c)
    tl.store(Y + base + 3, d)
    if tl.program_id(0) == 0:
        tl.store(Stamp + chunk, chunk + 1)


def launch(x, y, stamp, reps, chunk, ctas=8):
    return _aux_fp32[(ctas,)](x, y, stamp, reps, chunk, 128, num_warps=4)
