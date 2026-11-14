"""Tensor-core probes (v3): BF16 GEMM and native NVFP4 GEMM, with identity check.

The old probe was a bare `a @ b`; we could not say which kernel ran or whether it
used the tensor cores. This module:
  * builds either a BF16 or a native NVFP4 GEMM of the same tile count,
  * can report the exact CUDA kernel name that executes (torch.profiler),
  * keeps the operand footprint fixed so the probe is compute dominated.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _to_blocked(s):
    r, c = s.shape
    rb, cb = (r + 127) // 128, (c + 3) // 4
    p = np.zeros((rb * 128, cb * 4), np.float32); p[:r, :c] = s
    return p.reshape(rb, 128, cb, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3).reshape(-1)


class TensorProbe:
    """dtype: 'bf16' or 'fp4'. n: square operand size."""

    def __init__(self, dtype="bf16", n=2048, dev="cuda", seed=0):
        self.dtype = dtype
        self.n = n
        self.dev = dev
        rng = np.random.default_rng(seed)
        if dtype == "bf16":
            self.a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
            self.b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
            self.fn = lambda: self.a @ self.b
            self.meta = dict(kind="bf16", n=n, flops=2.0 * n ** 3)
        elif dtype == "fp4":
            import nvfp4_recipe as R
            self._R = R
            a = torch.from_numpy(rng.integers(0, 256, (n, n // 2), dtype=np.uint8)).to(dev).view(torch.float4_e2m1fn_x2)
            b = torch.from_numpy(rng.integers(0, 256, (n, n // 2), dtype=np.uint8)).to(dev).view(torch.float4_e2m1fn_x2)
            sa = torch.from_numpy(_to_blocked(rng.uniform(.01, .08, (n, n // 16)).astype(np.float32))).to(dev).to(torch.float8_e4m3fn).reshape(-1)
            sb = torch.from_numpy(_to_blocked(rng.uniform(.01, .08, (n, n // 16)).astype(np.float32))).to(dev).to(torch.float8_e4m3fn).reshape(-1)
            self.a, self.b, self.sa, self.sb = a, b, sa, sb
            self.fn = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            self.meta = dict(kind="fp4", n=n, flops=2.0 * n ** 3)
        else:
            raise ValueError(dtype)

    def __call__(self):
        return self.fn()

    def identify(self, reps=5):
        """Return the CUDA kernel name(s) that actually run."""
        for _ in range(3):
            self.fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                self.fn()
            torch.cuda.synchronize()
        rows = []
        for e in prof.key_averages():
            if e.device_time_total > 0:
                rows.append(dict(kernel=e.key, count=e.count, cuda_us=e.device_time_total))
        rows.sort(key=lambda r: -r["cuda_us"])
        self.meta["kernels"] = rows[:3]
        return rows


if __name__ == "__main__":
    for dt in ("bf16", "fp4"):
        p = TensorProbe(dt, n=2048)
        rows = p.identify()
        print(dt, "->", rows[0]["kernel"][:120] if rows else "NONE")
