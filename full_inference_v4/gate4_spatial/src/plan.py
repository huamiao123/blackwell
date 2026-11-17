"""Frozen, small first-wave plan. Produces plans, never performance observations."""
import argparse,random
from g4_common import *

MODES=['a_alone','b_alone','serial_ab','serial_ba','parallel_ab','parallel_ba',
       'empty_control','same_stream_control','non_overlap_control']
GRIDS=[16,32,64,128,170,192,256]

def probe_specs():
    # Grid-stride logical tiles keep total work fixed as Grid changes.
    return [dict(probe='fp32_fma',variant=v,threads=t,shared_bytes=s,items=262144,vector=16,rounds=512,
                 input_pattern='0.25 + ((16*i+j)%64)/128',expected_flops=262144*16*512*2)
            for v,t,s in [('light',32,0),('heavy',128,32768)]]+[
        dict(probe='bf16_mma',variant=v,threads=t,shared_bytes=s,tiles=4096,m=16,n=16,k_step=16,k_staging=k,
             mma_rounds=256,expected_flops=4096*256*2*16*16*16)
        for v,t,s,k in [('light',32,1024,16),('heavy',128,32768,128)]]

def make_plan(selection,seed=41023):
    specs=probe_specs()
    for s in specs:s['logical_work_hash']=digest(s)
    standalone=[dict(spec=s,grids=GRIDS,status='PLANNED_NOT_RUN') for s in specs]
    # First pilot only: QKV + FP32 light at 32/128 CTAs; broadening requires review.
    pilot=[dict(case=c,probe=specs[0],probe_grid=g) for c in selection['cases'] if c['operator']=='qkv_proj' for g in [32,128]]
    blocks=[]
    for b in range(5):
        order=['nvfp4','bf16'] if b%2==0 else ['bf16','nvfp4']
        conditions=[r for p in order for r in pilot if r['case']['precision']==p]
        for cond in conditions:
            modes=MODES[:];random.Random(seed+b*17+cond['probe_grid']).shuffle(modes)
            blocks.append(dict(block=b,condition=cond,modes=modes,profile='P0',status='PLANNED_NOT_RUN'))
    return dict(schema=1,cpu_preparation_only=True,gpu_authorized=False,standalone=standalone,pilot_blocks=blocks,
                later='Gate/Up, Tensor probe and Heavy configurations only after pilot correctness/timeline review',
                independent_process_recheck='required for key findings; separately authorized',
                trace_plan='Separate P2 runs, same logical work/config hashes; never pooled into P0')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:raise SystemExit('GPU_DISABLED_CPU_PREPARATION: no reviewed Gate-4 GPU window')
    save(SPATIAL/'configs/experiment_plan.json',make_plan(load(SPATIAL/'configs/kernel_selection.json')))
    print('CPU experiment plan saved; zero CUDA work')
