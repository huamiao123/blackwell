import torch, json
A=torch.randn(1024,4096,device='cuda',dtype=torch.bfloat16)
W=torch.randn(12288,4096,device='cuda',dtype=torch.bfloat16)
for _ in range(5): A@W.t()
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    for _ in range(5): A@W.t()
    torch.cuda.synchronize()
rows=[]
for e in p.key_averages():
    if e.device_time_total>0:
        rows.append(dict(count=e.count,cuda_us=round(e.device_time_total,1),kernel=e.key))
rows.sort(key=lambda r:-r['cuda_us'])
for r in rows[:5]: print(r['count'], r['cuda_us'], r['kernel'][:150])
json.dump(rows, open('/root/fp4_bf16_study/raw/bf16_kernels.json','w'), indent=2)