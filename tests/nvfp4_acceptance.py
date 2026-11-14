"""RTX 5090 native NVFP4 GEMM numerical acceptance (6 stages).

Every stage reports max_abs_error, relative_L2, cosine, P99, NaN/Inf and the
location of the worst element. Stage 1 is the gate: if the trivial all-ones case
fails, formal acceptance stops (diagnostic stages may still run).

NO GPU WORK HAPPENS UNLESS THIS FILE IS RUN MANUALLY.

Usage:
  .venv/bin/python tests/nvfp4_acceptance.py                 # all stages
  .venv/bin/python tests/nvfp4_acceptance.py --stages s1,s2  # subset
  .venv/bin/python tests/nvfp4_acceptance.py --json raw/acc.json
"""
import argparse, json, sys, time
import numpy as np
import torch
try:
    import torch._higher_order_ops.flex_gemm  # registers flex_gemm::to_blocked (torch>=2.15)
except Exception:  # older torch: we use our own to_blocked_np anyway
    pass
sys.path.insert(0, 'src')
import nvfp4_recipe as R   # corrected encode/decode scale split (FP32 global scale)

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
CODE = {float(v): i for i, v in enumerate(E2M1)}
CODE_NEG = {float(-v): i + 8 for i, v in enumerate(E2M1[1:])}
def val_to_code(v):
    i = int(np.argmin(np.abs(E2M1 - abs(v))))
    return (i | 8) if v < 0 else i
def nib_to_val(n):
    return E2M1[n & 7] * np.where(n & 8, -1.0, 1.0)
def pack(nib):
    return ((nib[..., 1::2] << 4) | nib[..., 0::2]).astype(np.uint8)

def to_blocked_np(s2d):
    rows, cols = s2d.shape
    rb, cb = (rows+127)//128, (cols+3)//4
    pad = np.zeros((rb*128, cb*4), np.float32); pad[:rows, :cols] = s2d
    return pad.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3).reshape(-1)

def round_e4m3(x):
    t = torch.tensor(np.asarray(x, np.float32)).to(torch.float8_e4m3fn)
    return t.float().numpy()

def metrics(o, ref):
    o = np.asarray(o, np.float64); ref = np.asarray(ref, np.float64)
    d = np.abs(o - ref)
    m = dict(max_abs_error=float(d.max()),
             rmse=float(np.sqrt((d**2).mean())),
             relative_L2=float(np.linalg.norm(o-ref)/max(np.linalg.norm(ref), 1e-30)),
             cosine=float((o*ref).sum()/max(np.linalg.norm(o)*np.linalg.norm(ref), 1e-30)),
             p99_abs_error=float(np.percentile(d, 99)),
             nan_inf=int(np.isnan(o).sum()+np.isinf(o).sum()))
    idx = np.unravel_index(int(np.argmax(d)), o.shape)
    m['worst_index'] = [int(i) for i in idx]
    m['worst_got'] = float(o[idx]); m['worst_ref'] = float(ref[idx])
    return m

def run_mm(a_nib, w_nib, sa, sb, alpha=1.0, dev="cuda"):
    """a_nib (M,K), w_nib (N,K), scales fp32 (M,kb),(N,kb) -> (out_fp32, meta)"""
    ta = torch.from_numpy(pack(a_nib)).to(dev).view(torch.float4_e2m1fn_x2)
    tw = torch.from_numpy(pack(w_nib)).to(dev).view(torch.float4_e2m1fn_x2)
    tsa = torch.from_numpy(to_blocked_np(sa)).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    tsb = torch.from_numpy(to_blocked_np(sb)).to(dev).to(torch.float8_e4m3fn).reshape(-1)
    out = torch._scaled_mm(ta, tw.t(), scale_a=tsa, scale_b=tsb, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    return out.float() * alpha

def ref_mm(a_nib, w_nib, sa, sb, alpha=1.0):
    A = nib_to_val(a_nib) * np.repeat(sa, 16, axis=1)
    W = nib_to_val(w_nib) * np.repeat(sb, 16, axis=1)
    return (A @ W.T) * alpha

def check(name, o, ref, kind, tol_max_abs=None, tol_rel=2e-2):
    o = o.cpu().numpy() if hasattr(o, 'cpu') else np.asarray(o)
    m = metrics(o, ref)
    if tol_max_abs is not None:  ok = m['max_abs_error'] <= tol_max_abs
    else:                        ok = m['relative_L2'] <= tol_rel
    m['stage'] = name; m['kind'] = kind; m['pass'] = bool(ok)
    print(f"  {name:<18} max_abs={m['max_abs_error']:.3e} rel_L2={m['relative_L2']:.3e} "
          f"cos={m['cosine']:.6f} p99={m['p99_abs_error']:.3e} "
          f"worst@{tuple(m['worst_index'])} {m['worst_got']:.4f}/{m['worst_ref']:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return m

M, K, N = 128, 256, 128
kb = K // 16
ones = np.full((M, K), CODE[1.0], np.uint8)
sa1 = np.ones((M, kb), np.float32); sb1 = np.ones((N, kb), np.float32)

def stage1():
    o = run_mm(ones, ones, sa1, sb1)
    ref = np.full((M, N), float(K), np.float32)
    return [check('s1_all_ones', o, ref, 'exact', tol_max_abs=1e-3)]

def stage2():
    out = []
    for pos in (0, 1, 16, 17, K-1):
        a = np.zeros((M, K), np.uint8); a[0, pos] = CODE[1.0]
        w = np.zeros((N, K), np.uint8); w[0, pos] = CODE[1.0]
        ref = np.zeros((M, N), np.float32); ref[0, 0] = 1.0
        out.append(check(f's2_onehot_k{pos}', run_mm(a, w, sa1, sb1), ref, 'exact', tol_max_abs=1e-3))
    return out

def stage3():
    out = []
    for kbi in (0, 5, kb-1):
        sa = np.ones((M, kb), np.float32); sa[0, kbi] = 2.0
        a = np.zeros((M, K), np.uint8); a[0, 16*kbi] = CODE[1.0]
        w = np.zeros((N, K), np.uint8); w[0, 16*kbi] = CODE[1.0]
        ref = np.zeros((M, N), np.float32); ref[0, 0] = 2.0
        out.append(check(f's3_blockscale_kb{kbi}', run_mm(a, w, sa, sb1), ref, 'exact', tol_max_abs=1e-3))
    return out

def stage4():
    sa = np.ones((M, kb), np.float32); sa[1, :] = 4.0; sa[7, :] = 0.5
    a = np.zeros((M, K), np.uint8)
    for r in (0, 1, 7): a[r, 0] = CODE[1.0]
    w = np.zeros((N, K), np.uint8); w[0, 0] = CODE[1.0]
    ref = np.zeros((M, N), np.float32); ref[0,0]=1.0; ref[1,0]=4.0; ref[7,0]=0.5
    return [check('s4_row_scale', run_mm(a, w, sa, sb1), ref, 'exact', tol_max_abs=1e-3)]

def stage5():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 16, (M, K)).astype(np.uint8)
    w = rng.integers(0, 16, (N, K)).astype(np.uint8)
    sa = round_e4m3(rng.uniform(.01, .1, (M, kb)))
    sb = round_e4m3(rng.uniform(.01, .1, (N, kb)))
    return [check('s5_random', run_mm(a, w, sa, sb), ref_mm(a, w, sa, sb), 'bf16_out')]

def stage6():
    """Same-source BF16 -> NVFP4, compared against the quantised FP32 reference.

    Recipe: g_encode = 2688/amax (FP32, never E4M3), block scale E4M3 per 16 elements,
    decode by multiplying the raw GEMM by 1/(g_a*g_b)."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    A[rng.random(A.shape) < 0.02] *= 8.0                     # outliers
    W = rng.standard_normal((N, K)).astype(np.float32)
    qa = R.quantize(A); qb = R.quantize(W)
    alpha = R.gemm_alpha(qa['g_encode'], qb['g_encode'])
    o = run_mm(qa['payload'], qb['payload'], qa['block_scale'], qb['block_scale'], alpha=alpha)
    ref = ref_mm(qa['payload'], qb['payload'], qa['block_scale'], qb['block_scale'], alpha=alpha)
    m = check('s6_same_source_bf16', o, ref, 'bf16_out')
    m['g_encode_a'] = qa['g_encode']; m['g_encode_b'] = qb['g_encode']
    m['alpha'] = alpha
    m['quant_roundtrip_rel_L2_A'] = float(np.linalg.norm(qa['dequant'] - A) / np.linalg.norm(A))
    return [m]

STAGES = [('s1', stage1), ('s2', stage2), ('s3', stage3), ('s4', stage4), ('s5', stage5), ('s6', stage6)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='s1,s2,s3,s4,s5,s6')
    ap.add_argument('--json', default='raw/nvfp4_acceptance.json')
    args = ap.parse_args()
    want = [s.strip() for s in args.stages.split(',')]
    if not torch.cuda.is_available():
        print('no CUDA device'); return 2
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
          f"scale numel formula: 128*ceil(rows/128)*ceil(kb/4)*4")
    results, gate_failed = [], False
    for sid, fn in STAGES:
        if sid not in want: continue
        if sid == 's1':
            print('=== stage 1: all FP4=1, all scales=1 (GATE) ===')
            r = fn(); results += r
            if not all(x['pass'] for x in r):
                gate_failed = True
                print('GATE FAILED: trivial all-ones case is wrong -> stop formal acceptance')
                break
        else:
            print(f'=== stage {sid[1]} ===')
            try: results += fn()
            except Exception as e:
                print(f'  {sid} raised {type(e).__name__}: {e}')
                results.append(dict(stage=sid, pass_=False, error=str(e)))
    ok = (not gate_failed) and all(x.get('pass') for x in results)
    print()
    print(f"SUMMARY: {sum(1 for x in results if x.get('pass'))}/{len(results)} checks passed")
    print('ACCEPTANCE:', 'PASS' if ok else 'NOT PASSED')
    json.dump({'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
               'when': time.strftime('%Y-%m-%dT%H:%M:%S'), 'results': results,
               'acceptance': 'PASS' if ok else 'NOT_PASSED'},
              open(args.json, 'w'), indent=2)
    print('json ->', args.json)
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
