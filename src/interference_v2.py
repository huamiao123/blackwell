"""Interference matrix v2: duration-matched, 3 pressure levels, 3 sessions.

Protocol fixes over v1 (V2 task book §6.3 / §8):
  * probe work is calibrated so T_B ~= k * T_A  (k = 0.5, 1, 2)
  * both "same-work" (k sweep, probe work varies with fixed main work) and
    duration-matched points are recorded
  * 3 independent sessions, median + min/max reported
  * GPU power / SM clock sampled around every condition (power-wall control)
  * all timings in milliseconds, clearly labelled
"""
import json, subprocess, sys
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

def probe_fp32(dev='cuda'):
    n = 128 * 1024 * 1024 // 4
    x = torch.ones(n, device=dev, dtype=torch.float32); y = torch.ones(n, device=dev, dtype=torch.float32)
    return lambda: torch.add(x, y, out=x)

def probe_dram(dev='cuda'):
    n = 1536 * 1024 * 1024 // 4
    x = torch.ones(n, device=dev, dtype=torch.float32)
    return lambda: torch.sum(x)

def probe_tensor(dev='cuda', n=2048):
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16); b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    return lambda: a @ b

def gpu_state():
    try:
        o = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw,clocks.sm,temperature.gpu',
                                     '--format=csv,noheader,nounits'], timeout=5).decode().strip()
        p, c, t = [float(v) for v in o.split(',')]
        return dict(power_w=p, sm_mhz=c, temp_c=t)
    except Exception:
        return dict(power_w=None, sm_mhz=None, temp_c=None)

def single_ms(fn, warm=3, n=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n

def timed(fn, reps):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)

def concurrent(fA, rA, fB, rB):
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    ctrl = torch.cuda.current_stream()
    ready = torch.cuda.Event(True); a0 = torch.cuda.Event(True); a1 = torch.cuda.Event(True)
    b0 = torch.cuda.Event(True); b1 = torch.cuda.Event(True); joint = torch.cuda.Event(True)
    torch.cuda.synchronize(); ready.record(ctrl)
    sA.wait_event(ready); sB.wait_event(ready)
    with torch.cuda.stream(sA):
        a0.record(sA)
        for _ in range(rA): fA()
        a1.record(sA)
    with torch.cuda.stream(sB):
        b0.record(sB)
        for _ in range(rB): fB()
        b1.record(sB)
    ctrl.wait_event(a1); ctrl.wait_event(b1); joint.record(ctrl); joint.synchronize()
    return a0.elapsed_time(a1), b0.elapsed_time(b1), ready.elapsed_time(joint)

def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    N, K = 12288, 4096
    probes = dict(fp32=probe_fp32, dram=probe_dram, tensor=probe_tensor)
    ks = (0.5, 1.0, 2.0)
    rows = []
    hdr = f"{'M':>5} {'dt':>5} {'probe':>6} {'k':>4} {'rA':>4} {'rB':>5} {'T_Ams':>7} {'T_Bms':>7} {'Ser':>7} {'Joint':>7} {'S_A':>5} {'S_B':>5} {'G':>5} {'W':>6} {'MHz':>5}"
    print(hdr); print('-' * len(hdr))
    for M in (16, 128, 1024):
        mains = dict(bf16=make_bf16(M, N, K), fp4=make_fp4(M, N, K, rng))
        for dtype, fA in mains.items():
            tA1 = single_ms(fA)
            rA = max(5, int(round(20.0 / tA1)))
            tA = timed(fA, rA)
            for pname, pfac in probes.items():
                fB = pfac(); tB1 = single_ms(fB)
                for k in ks:
                    rB = max(2, int(round(k * tA / tB1)))
                    samples = []
                    for sess in range(3):
                        tB = timed(fB, rB)
                        tSer = timed(fA, rA) + tB
                        st = gpu_state()
                        ca, cb, tj = concurrent(fA, rA, fB, rB)
                        samples.append(dict(T_A=ca, T_B=cb, T_ser=tSer, T_joint=tj,
                                            S_A=(ca / rA) / (tA / rA), S_B=(cb / rB) / (tB / rB),
                                            G=tSer / tj, **st))
                    med = lambda key: float(np.median([s[key] for s in samples]))
                    r = dict(M=M, dtype=dtype, probe=pname, k=k, reps_A=rA, reps_B=rB,
                             T_A_ms=med('T_A'), T_B_ms=med('T_B'), T_serial_ms=med('T_ser'),
                             T_joint_ms=med('T_joint'), S_A=med('S_A'), S_B=med('S_B'), G=med('G'),
                             S_A_min=min(s['S_A'] for s in samples), S_A_max=max(s['S_A'] for s in samples),
                             power_w=med('power_w'), sm_mhz=med('sm_mhz'), temp_c=med('temp_c'))
                    rows.append(r)
                    print(f"{M:5d} {dtype:>5} {pname:>6} {k:4.1f} {rA:4d} {rB:5d} {r['T_A_ms']:7.2f} "
                          f"{r['T_B_ms']:7.2f} {r['T_serial_ms']:7.2f} {r['T_joint_ms']:7.2f} "
                          f"{r['S_A']:5.2f} {r['S_B']:5.2f} {r['G']:5.2f} "
                          f"{(r['power_w'] or 0):6.0f} {(r['sm_mhz'] or 0):5.0f}")
        del mains; torch.cuda.empty_cache()
    json.dump(rows, open('raw/interference_v2.json', 'w'), indent=2)
    print('\njson -> raw/interference_v2.json  rows=', len(rows))

if __name__ == '__main__':
    main()
