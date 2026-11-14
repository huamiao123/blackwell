import json, torch, numpy as np
def blocked(s):
    r,c=s.shape; rb,cb=(r+127)//128,(c+3)//4
    p=np.zeros((rb*128,cb*4),np.float32); p[:r,:c]=s
    return p.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3).reshape(-1)
def t(fn,w=10,n=30,i=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); ts=[]
    for _ in range(n):
        e0.record()
        for _ in range(i): fn()
        e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1)*1000/i)
    return sorted(ts)[len(ts)//2]
rng=np.random.default_rng(0); out=[]
for M in (16,128,512,1024):
  for N,K in ((12288,4096),(4096,4096)):
    A=torch.randn(M,K,device='cuda',dtype=torch.bfloat16); W=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
    tb=t(lambda: A@W.t())
    ta=torch.from_numpy(rng.integers(0,256,(M,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
    tw=torch.from_numpy(rng.integers(0,256,(N,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
    sa=torch.from_numpy(blocked(rng.uniform(.01,.08,(M,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
    sb=torch.from_numpy(blocked(rng.uniform(.01,.08,(N,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
    tf=t(lambda: torch._scaled_mm(ta,tw.t(),scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16))
    f=2.0*M*N*K
    bb=M*K*2+N*K*2+M*N*2; bf=M*K*.5+N*K*.5+N*(K//16)+M*(K//16)+M*N*2
    r=dict(M=M,N=N,K=K,bf16_us=tb,fp4_us=tf,bf16_tf=f/tb/1e6,fp4_tf=f/tf/1e6,
           speedup=tb/tf,bf16_gbs=bb/tb/1e3,fp4_gbs=bf/tf/1e3)
    out.append(r)
    print(f"M={M:4d} N={N:5d} K={K}: BF16 {tb:8.1f}us {r['bf16_tf']:6.1f}TF {r['bf16_gbs']:6.0f}GB/s | FP4 {tf:8.1f}us {r['fp4_tf']:6.1f}TF {r['fp4_gbs']:6.0f}GB/s | x{r['speedup']:.2f}")
json.dump(out,open('/root/fp4_bf16_study/raw/bench_precision_first.json','w'),indent=2)