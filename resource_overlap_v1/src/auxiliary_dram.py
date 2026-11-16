"""Independent 256 MiB source and destination; full streaming copy per pass."""
import triton
import triton.language as tl


@triton.jit
def _aux_dram(X, Y, Stamp, N: tl.constexpr, CHUNK, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for base in range(tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK):
        i = base + lane
        v = tl.load(X + i, i < N, 0, cache_modifier=".cg")
        tl.store(Y + i, v, i < N, cache_modifier=".wb")
    if tl.program_id(0) == 0:
        tl.store(Stamp + CHUNK, CHUNK + 1)


def launch(x, y, stamp, chunk):
    return _aux_dram[(256,)](x, y, stamp, x.numel(), chunk, 1024, num_warps=4)
