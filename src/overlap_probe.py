import json, numpy as np, torch
def union(iv):
    iv=sorted(iv); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]: out[-1][1]=max(out[-1][1],e)
        else: out.append([s,e])
    return out
def length(iv): return sum(e-s for s,e in iv)
def inter(a,b):
    a=union(a); b=union(b); i=j=0; tot=0
    while i<len(a) and j<len(b):
        s=max(a[i][0],b[j][0]); e=min(a[i][1],b[j][1])
        if e>s: tot+=e-s
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return tot
M,N,K=256,12288,4096
rng=np.random.default_rng(0)
def blocked(s):
    r,c=s.shape; rb,cb=(r+127)//128,(c+3)//4
    p=np.zeros((rb*128,cb*4),np.float32); p[:r,:c]=s
    return p.reshape(rb,128,cb,4).transpose(0,2,1,3).reshape(-1,4,32,4).transpose(0,2,1,3).reshape(-1)
ta=torch.from_numpy(rng.integers(0,256,(M,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
tw=torch.from_numpy(rng.integers(0,256,(N,K//2),dtype=np.uint8)).cuda().view(torch.float4_e2m1fn_x2)
sa=torch.from_numpy(blocked(rng.uniform(.01,.08,(M,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
sb=torch.from_numpy(blocked(rng.uniform(.01,.08,(N,K//16)).astype(np.float32))).cuda().to(torch.float8_e4m3fn).reshape(-1)
fA=lambda: torch._scaled_mm(ta,tw.t(),scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16)
a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16); b=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)
fB=lambda: a@b
for _ in range(5): fA(); fB()
torch.cuda.synchronize()
sA,sB=torch.cuda.Stream(),torch.cuda.Stream(); ctrl=torch.cuda.current_stream()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    ready=torch.cuda.Event(); ready.record(ctrl); sA.wait_event(ready); sB.wait_event(ready)
    with torch.cuda.stream(sA):
        for _ in range(60): fA()
    with torch.cuda.stream(sB):
        for _ in range(400): fB()
    ctrl.wait_stream(sA); ctrl.wait_stream(sB); torch.cuda.synchronize()
prof.export_chrome_trace('/root/fp4_bf16_study/raw/trace_concurrent.json')
ev=json.load(open('/root/fp4_bf16_study/raw/trace_concurrent.json'))['traceEvents']
ks=[e for e in ev if e.get('cat')=='kernel']
streams=sorted({e['args'].get('stream') for e in ks})
by={}
for e in ks: by.setdefault(e['args'].get('stream'),[]).append((e['ts'],e['ts']+e['dur']))
print('kernels:', len(ks), 'streams:', streams)
info={}
for s,iv in by.items():
    info[s]=dict(n=len(iv), busy_us=length(union(iv)))
print('per-stream:', {s:(v['n'],round(v['busy_us'],1)) for s,v in info.items()})
ids=sorted(by)
if len(ids)>=2:
    A=by[ids[0]]; B=by[ids[1]]
    ov=inter(A,B)
    print(f'stream {ids[0]} busy={length(union(A)):.1f}us, stream {ids[1]} busy={length(union(B)):.1f}us, overlap={ov:.1f}us')
    print(f'overlap/min = {ov/min(length(union(A)),length(union(B)))*100:.1f}%')
json.dump({str(s):dict(n=v['n'],busy_us=v['busy_us']) for s,v in info.items()}, open('/root/fp4_bf16_study/raw/overlap.json','w'), indent=2)