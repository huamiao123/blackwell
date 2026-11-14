"""Dual-stream interference matrix: BF16 vs native NVFP4 main GEMM against
resource-specific probes. No profiler required - only CUDA events.

Metrics per (M, dtype, probe):
  S_A = T_A_conc / T_A_alone     main-task slowdown
  S_B = T_B_conc / T_B_alone     probe slowdown
  G   = T_serial / T_joint       joint gain (>1 means concurrency helps)

Interpretation: a probe that slows the main task down is competing for a
*scarce* resource; a probe that costs the main task nothing is exploiting
*spare* capacity. Comparing BF16 vs FP4 on the same probe is the measurement.
"""
import json, sys
import numpy as np, torch

def blocked(s):
    r, c = s.shape
    rb, cb = (r + 127) // 128, (c + 3) // 4
    p = np.zeros((rb * 128, cb * 4), np.float32); p[:r, :c] = s
    return p.reshape(rb, 128, cb, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(0, 2, 1, 3).reshape(-1)

def make_fp4(M, N, K, rng, dev='cuda'):
    ta = torch.from_numpy(rng.integers(0, 256, (M, K // 2), dtype=np.uint8)).to(dev).view(torch.float4_e2m1fn_x2)
    tw = torch.from_numpy(rng.integers(0, 256, (N, K // 2), dtype=np.uint8)).to(dev).view(torch.float4_e2m1fn_x2)
    sa = torch.from_numpy(blocked(rng.uniform(.01, .08, (M, K // 16)).astype(np.float32))).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    sb = torch.from_numpy(blocked(rng.uniform(.01, .08, (N, K // 16)).astype(np.float32))).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    return lambda: torch._scaled_mm(ta, tw.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

def make_bf16(M, N, K, dev='cuda'):
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    return lambda: A @ W.t()

def probe_fp32(dev='cuda', mb=32):
    n = mb * 1024 * 1024 // 4
    x = torch.ones(n, device=dev, dtype=torch.float32)
    y = torch.ones(n, device=dev, dtype=torch.float32)
    return lambda: torch.add(x, y, out=x)

def probe_dram(dev='cuda', mb=1536):
    n = mb * 1024 * 1024 // 4
    x = torch.ones(n, device=dev, dtype=torch.float32)
    return lambda: torch.sum(x)

def probe_tensor(dev='cuda', n=2048):
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    return lambda: a @ b

def time_alone(fn, reps, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)

def time_serial(fA, rA, fB, rB):
    for _ in range(2): fA(); fB()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(rA): fA()
    for _ in range(rB): fB()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)

def time_concurrent(fA, rA, fB, rB):
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    ctrl = torch.cuda.current_stream()
    ready = torch.cuda.Event(); a0 = torch.cuda.Event(True); a1 = torch.cuda.Event(True)
    b0 = torch.cuda.Event(True); b1 = torch.cuda.Event(True); joint = torch.cuda.Event(True)
    torch.cuda.synchronize()
    ready.record(ctrl)
    sA.wait_event(ready); sB.wait_event(ready)
    with torch.cuda.stream(sA):
        a0.record(sA)
        for _ in range(rA): fA()
        a1.record(sA)
    with torch.cuda.stream(sB):
        b0.record(sB)
        for _ in range(rB): fB()
        b1.record(sB)
    ctrl.wait_event(a1); ctrl.wait_event(b1)
    joint.record(ctrl); joint.synchronize()
    return a0.elapsed_time(a1), b0.elapsed_time(b1), ready.elapsed_time(joint)

def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, K = 12288, 4096
    probes = dict(fp32=probe_fp32, dram=probe_dram, tensor=probe_tensor)
    probe_reps = dict(fp32=300, dram=20, tensor=200)
    rows = []
    print(f"{'M':>5} {'dtype':>5} {'probe':>7} {'T_A':>8} {'T_B':>8} {'T_serial':>9} {'T_joint':>8} "
          f"{'S_A':>6} {'S_B':>6} {'G':>6}")
    for M in (16, 128, 1024):
        mains = dict(bf16=make_bf16(M, N, K), fp4=make_fp4(M, N, K, rng))
        main_reps = dict(bf16=40, fp4=200)
        for dtype, fA in mains.items():
            rA = main_reps[dtype]
            ta = time_alone(fA, rA) / rA
            for pname, pfac in probes.items():
                fB = pfac()
                rB = probe_reps[pname]
                tb = time_alone(fB, rB) / rB
                ts = time_serial(fA, rA, fB, rB)
                ca, cb, tj = time_concurrent(fA, rA, fB, rB)
                S_A = (ca / rA) / ta
                S_B = (cb / rB) / tb
                G = ts / tj
                rows.append(dict(M=M, dtype=dtype, probe=pname,
                                 T_A_alone_us=ta, T_B_alone_us=tb,
                                 T_serial_ms=ts, T_joint_ms=tj,
                                 S_A=S_A, S_B=S_B, G=G, reps_A=rA, reps_B=rB))
                print(f"{M:5d} {dtype:>5} {pname:>7} {ta:8.1f} {tb:8.2f} {ts:9.2f} {tj:8.2f} "
                      f"{S_A:6.2f} {S_B:6.2f} {G:6.2f}")
        del mains
        torch.cuda.empty_cache()
    json.dump(rows, open('raw/interference_matrix.json', 'w'), indent=2)
    print('\njson -> raw/interference_matrix.json')

if __name__ == '__main__':
    main()
