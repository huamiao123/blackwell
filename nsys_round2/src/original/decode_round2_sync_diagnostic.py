"""Real LLM.generate requests, using the successful decode_prof2 launch path.

Instrumentation observes engine steps/scheduled requests; no model/kernel edits.
The first prompt-producing step is excluded from the 256 decode steps.
"""
import argparse,os,json,time,hashlib,statistics,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--precision',choices=['bf16','nvfp4'],required=True)
    ap.add_argument('--mode',choices=['eager','graph'],required=True)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--kv',type=int,default=2048)
    ap.add_argument('--steps',type=int,default=256)
    ap.add_argument('--trials',type=int,default=5)
    ap.add_argument('--profile',action='store_true')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    import torch
    from vllm import LLM,SamplingParams
    p=Path.home()/'fp4-llm'
    outpath=Path(a.output)
    assert outpath.is_relative_to(p) and not outpath.exists()
    ids=[1000+(i%5000) for i in range(a.kv)]
    prompts=[{'prompt_token_ids':ids.copy()} for _ in range(a.batch)]
    settings=dict(max_model_len=4096,gpu_memory_utilization=.85,
        linear_backend='cutlass',enforce_eager=a.mode=='eager',
        kv_cache_dtype='fp8_e4m3',attention_backend='FLASHINFER',
        disable_log_stats=True,enable_prefix_caching=True,
        max_num_batched_tokens=8192,max_num_seqs=8,
        cudagraph_capture_sizes=[1,8],async_scheduling=False,seed=0)
    model=p/'models'/('Qwen3-8B-BF16' if a.precision=='bf16' else 'Qwen3-8B-NVFP4')
    llm=LLM(model=str(model),**settings)
    eng=llm.llm_engine
    core=eng.engine_core.engine_core
    assert type(eng.engine_core).__name__=='InprocClient'
    assert core.async_scheduling is False
    print('ROUND2_MODEL_READY',os.getpid(),a.precision,a.mode,flush=True)
    result=dict(args=vars(a),pid=os.getpid(),engine_core_pid=os.getpid(),
        engine_client=type(eng.engine_core).__name__,settings=settings,
        model=str(model),torch_version=torch.__version__,
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        device=dict(name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability()),
        cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
        scheduler_type=type(core.scheduler).__name__,runs=[])
    original_step=eng.step
    original_schedule=core.scheduler.schedule
    original_replay=torch.cuda.CUDAGraph.replay
    state={'measured':False,'outer':0,'replays':0,'schedules':[],'steps':[]}
    def replay_hook(self,*args,**kwargs):
        if state['measured']:state['replays']+=1
        return original_replay(self,*args,**kwargs)
    torch.cuda.CUDAGraph.replay=replay_hook
    def schedule_hook(*args,**kwargs):
        if state['measured']:torch.cuda.nvtx.range_push('CPU_SCHEDULER')
        try:s=original_schedule(*args,**kwargs)
        finally:
            if state['measured']:torch.cuda.nvtx.range_pop()
        if state['measured']:
            reqs=[core.scheduler.requests[r] for r in s.num_scheduled_tokens]
            row=dict(outer_step=state['outer'],batch=len(reqs),
                     scheduled_tokens=list(s.num_scheduled_tokens.values()),
                     output_lengths=[r.num_output_tokens for r in reqs],
                     computed_tokens=[r.num_computed_tokens for r in reqs],
                     num_preempted=len(s.preempted_req_ids) if hasattr(s,'preempted_req_ids') else None)
            state['schedules'].append(row)
        return s
    core.scheduler.schedule=schedule_hook
    def step_hook(*args,**kwargs):
        if not state['measured']:return original_step(*args,**kwargs)
        ix=state['outer']
        if ix==1:
            torch.cuda.synchronize()
            if a.profile:torch.cuda.profiler.start()
            torch.cuda.nvtx.range_push('ROUND2_DECODE_WINDOW')
            state['window_start']=time.perf_counter_ns()
        if ix>=1:torch.cuda.nvtx.range_push(f'DECODE_STEP_{ix:04d}')
        r0=state['replays'];t0=time.perf_counter_ns()
        res=original_step(*args,**kwargs)
        t1=time.perf_counter_ns()
        if ix>=1:
            torch.cuda.nvtx.range_pop()
            state['steps'].append(dict(step=ix,start_ns=t0,end_ns=t1,latency_ms=(t1-t0)/1e6,replays=state['replays']-r0))
        if ix==a.steps:
            torch.cuda.synchronize()
            state['window_end']=time.perf_counter_ns()
            torch.cuda.nvtx.range_pop()
        state['outer']+=1
        return res
    eng.step=step_hook
    warm=SamplingParams(max_tokens=33,temperature=0.,ignore_eos=True,seed=0)
    for _ in range(2):llm.generate(prompts,warm,use_tqdm=False)
    torch.cuda.synchronize()
    for trial in range(a.trials):
        state.update(measured=True,outer=0,replays=0,schedules=[],steps=[])
        t0=time.perf_counter_ns()
        outputs=llm.generate(prompts,SamplingParams(max_tokens=a.steps+1,temperature=0.,ignore_eos=True,seed=0),use_tqdm=False)
        torch.cuda.synchronize(); t1=time.perf_counter_ns()
        state['measured']=False
        counts=[len(o.outputs[0].token_ids) for o in outputs]
        assert len(counts)==a.batch and counts==[a.steps+1]*a.batch,counts
        assert state['outer']==a.steps+1,state['outer']
        rows=[r for r in state['schedules'] if r['outer_step']>=1]
        assert len(rows)==a.steps and len(state['steps'])==a.steps
        assert all(r['batch']==a.batch and r['scheduled_tokens']==[1]*a.batch and min(r['output_lengths'])>=1 for r in rows),rows[:2]
        assert (state['replays']==0) if a.mode=='eager' else (state['replays']>=a.steps)
        wall=(state['window_end']-state['window_start'])/1e6
        lat=sorted(r['latency_ms'] for r in state['steps'])
        result['runs'].append(dict(trial=trial,decode_steps=a.steps,actual_batches=sorted({r['batch'] for r in rows}),
            decode_wall_ms=wall,mean_step_ms=wall/a.steps,p50_step_ms=statistics.median(lat),
            p95_step_ms=lat[int(.95*(len(lat)-1))],output_tokens_per_s=a.batch*a.steps/(wall/1000),
            generate_call_wall_ms=(t1-t0)/1e6,output_counts=counts,graph_replays=state['replays'],
            steps=state['steps'],scheduler=state['schedules'],
            generated_token_sha256=hashlib.sha256(json.dumps([o.outputs[0].token_ids for o in outputs]).encode()).hexdigest()))
        print('ROUND2_TRIAL',trial,wall,wall/a.steps,'ms_per_step','replays',state['replays'],flush=True)
    outpath.write_text(json.dumps(result,indent=2))
    print('ROUND2_SAVED',outpath,flush=True)
    if a.profile:torch.cuda.profiler.stop()
    eng.step=original_step;core.scheduler.schedule=original_schedule
    torch.cuda.CUDAGraph.replay=original_replay
    eng.engine_core.shutdown()

if __name__=='__main__':main()
