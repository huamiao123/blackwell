# Hazard register (living document)

| # | Hazard | Impact | Status / mitigation |
|---|---|---|---|
| H1 | Another user's compute job on the shared GPU | timing and L2 data silently polluted; power wall | `scripts/gpu_guard.sh` (exit 3 = busy). Formal runs MUST gate on it. |
| H2 | `/home` has no quota and no backup | data loss | `scripts/snapshot_results.sh`, keeps 5 newest tarballs; also pull to local disk |
| H3 | NCU hardware counters denied (`RmProfilingAdminOnly=1`) | no tensor-pipe/DRAM/stall attribution | blocked on admin: request `NOPASSWD: /usr/local/cuda/bin/ncu *`. Do NOT reload the driver on a shared box |
| H4 | CUTLASS 79a cannot initialise on SM120 | 79a abandoned | replaced by torch `_scaled_mm` NVFP4 path; layout root-caused (SWIZZLE_32_4_4), CPU-side verified; GPU acceptance pending approved window |
| H5 | Wrong python interpreter (system vs venv) | irreproducible results | always call `.venv/bin/python` explicitly; scripts do not rely on PATH |
| H6 | PyPI official index is ~0.07 MB/s here | multi-hour installs | RESOLVED: use `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` (measured 13 MB/s) |
| H7 | `nvidia-smi` wrapper rejects all arguments | env/monitoring commands fail silently | use `sudo -n /usr/bin/nvidia-smi` (account has NOPASSWD for that exact command) |
| H8 | GitHub unreachable from server | cannot clone CUTLASS / fetch models | download locally, scp up (measured ~0.5 MB/s); record tag+sha256 |
| H9 | CUDA 13.3 toolkit vs 595.99 driver (advertises 13.2) | minor-version compatibility not guaranteed for new APIs | observed OK for plain kernels; keep CUTLASS builds on this combo, re-test if switching |
| H10 | NVFP4 scale swizzle layout | would invalidate any comparison | RESOLVED: numpy `to_blocked` == torch `to_blocked` on 10/10 shapes, max_abs_diff = 0 |
| H11 | 30.9 GiB RAM shared with 5 users | OOM when loading 7B + quantising | keep `-j2` for compiles; do 7B only in an exclusive window |
