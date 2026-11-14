"""DRAM streaming probe (v3).

Replaces `torch.sum(x)`, whose reduction adds compute, a temporary and an
implementation-specific kernel hierarchy.

Design:
  * contiguous, coalesced, vectorised read+write (x -> out * 2)
  * arithmetic intensity ~1 FMA per 8 bytes -> bandwidth dominated
  * working set configurable; caller is expected to use >> L2 (96 MiB here)
  * bytes moved are known exactly: 2 * 4 * N (one read + one write)

Caveat that the caller must not forget: "bigger than L2" does NOT prove every
access comes from DRAM. Only a working-set sweep (see `sweep_bytes`) plus an
effective-bandwidth plateau gives indirect evidence.
"""
import triton
import triton.language as tl


@triton.jit
def _stream_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    v = tl.load(X + offs, mask=m)
    tl.store(OUT + offs, v * 2.0, mask=m)


class DramProbe:
    def __init__(self, mb=1536, block=4096, num_warps=8, dev="cuda"):
        import torch
        self.dev = dev
        n = int(mb) * 1024 * 1024 // 4
        n = (n // block) * block
        self.n = n
        self.block = block
        self.num_warps = num_warps
        self.x = torch.full((n,), 1.0, device=dev, dtype=torch.float32)
        self.out = torch.zeros_like(self.x)
        self.meta = dict(mb=mb, elements=n, bytes_read=4 * n, bytes_written=4 * n,
                         bytes_total=8 * n, block=block)

    def __call__(self):
        grid = (triton.cdiv(self.n, self.block),)
        return _stream_kernel[grid](self.x, self.out, self.n, BLOCK=self.block,
                                    num_warps=self.num_warps)

    def effective_gbps(self, ms):
        return self.meta["bytes_total"] / (ms * 1e-3) / 1e9


def build_dram(**kw):
    return DramProbe(**kw)


def sweep_bytes(sizes_mb=(16, 32, 64, 96, 128, 256, 512, 1024, 1536), repeats=5):
    """Working-set sweep: effective bandwidth vs footprint.

    A flat plateau above ~L2 is the standard evidence that we are DRAM bound;
    a step at 96 MiB reveals the L2 boundary. Returns a list of dicts.
    """
    import torch
    out = []
    for mb in sizes_mb:
        p = DramProbe(mb=mb)
        for _ in range(3):
            p()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(repeats):
            p()
        e1.record(); torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / repeats
        out.append(dict(mb=mb, ms=ms, gbps=p.effective_gbps(ms)))
        del p
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    for r in sweep_bytes():
        print(f"{r['mb']:5d} MiB  {r['ms']:7.3f} ms  {r['gbps']:7.1f} GB/s")
