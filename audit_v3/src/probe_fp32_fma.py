"""FP32 FMA probe (v3). Replaces the old `torch.add` elementwise add, which was
memory-bound and could not pressure the FP32 pipeline.

Design:
  * one element per thread, 4 independent accumulator chains (ILP=4) held in registers
  * no loads inside the loop -> arithmetic intensity is as high as we can get
  * a final store forces the compiler to keep the loop (no dead-code elimination)
  * grid size and REPS are independent knobs: grid controls how many SMs/CTAs are
    occupied, REPS controls instruction count per thread
  * the generated PTX is checked for `fma.rn.f32` so we know what we are measuring

Usage:
    from probe_fp32_fma import build_fma
    p = build_fma(n_blocks=170, block=256, reps=20000)
    p()                      # one launch
    p.meta                   # dict: blocks, block, reps, fma_per_launch, ptx_has_fma
"""
import triton
import triton.language as tl


@triton.jit
def _fma_kernel(X, OUT, REPS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + offs)
    # 4 independent chains -> ILP 4, all in registers
    x0 = a
    x1 = a * 1.0000001
    x2 = a * 1.0000002
    x3 = a * 1.0000003
    for _ in range(REPS):
        x0 = x0 * 0.9999999 + 0.0000001
        x1 = x1 * 0.9999998 + 0.0000002
        x2 = x2 * 0.9999997 + 0.0000003
        x3 = x3 * 0.9999996 + 0.0000004
    tl.store(OUT + offs, x0 + x1 + x2 + x3)


class FmaProbe:
    def __init__(self, n_blocks=170, block=256, reps=20000, num_warps=8, dev="cuda"):
        import torch
        self.dev = dev
        self.n_blocks = n_blocks
        self.block = block
        self.reps = reps
        self.num_warps = num_warps
        n = n_blocks * block
        self.x = torch.full((n,), 1.0, device=dev, dtype=torch.float32)
        self.out = torch.zeros_like(self.x)
        self.meta = dict(n_blocks=n_blocks, block=block, reps=reps,
                         threads=n_blocks * block * 4,          # 4 chains per lane
                         fma_per_launch=n * 4 * reps)
        self._compiled = None

    def __call__(self):
        return _fma_kernel[(self.n_blocks,)](self.x, self.out, self.reps,
                                             BLOCK=self.block, num_warps=self.num_warps)

    def verify_codegen(self):
        """Force a compile and inspect the PTX: must contain fma.rn.f32."""
        self()
        import torch
        torch.cuda.synchronize()
        k = _fma_kernel.warmup(self.x, self.out, self.reps, BLOCK=self.block,
                               grid=(1,), num_warps=self.num_warps)
        ptx = k.asm["ptx"]
        # sass may also be available depending on triton build
        sass = k.asm.get("sass", "")
        self.meta["ptx_has_fma"] = "fma.rn.f32" in ptx
        self.meta["ptx_fma_count"] = ptx.count("fma.rn.f32")
        self.meta["sass_has_ffma"] = ("FFMA" in sass) if sass else None
        self.meta["ptx_lines"] = ptx.count("\n")
        return self.meta


def build_fma(**kw):
    return FmaProbe(**kw)


if __name__ == "__main__":
    p = build_fma()
    m = p.verify_codegen()
    print(m)
