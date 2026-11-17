"""Prepared GPU worker. Dry run is CPU-only; execution requires supervisor permit.

No model is loaded on import. GPU integration remains UNVERIFIED until Gate 2.
"""
import argparse
import contextlib
import inspect
import importlib.metadata
import os
from pathlib import Path
import threading
import time
from common import ROOT,load,digest,file_hash,save,validate_config
from trace_state import TraceState


def prepare(config_path):
    c=load(config_path);expected=c.pop('config_hash')
    if digest(c)!=expected:raise ValueError('configuration hash mismatch')
    c['config_hash']=expected
    dataset=load(ROOT/'prompts/manifests/dataset.json')
    if digest(dataset['records'])!=c['dataset_manifest_sha256']:raise ValueError('dataset mismatch')
    entries={r['path']:r for r in dataset['records']}
    for f in c['prompt_paths']:
        if f not in entries or not (ROOT/f).resolve().is_relative_to(ROOT.resolve()):raise ValueError('prompt outside dataset')
        if file_hash(ROOT/f)!=entries[f]['file_sha256']:raise ValueError('prompt file hash mismatch')
    ps=[load(ROOT/f) for f in c['prompt_paths']]
    validate_config(c,ps)
    contract=load(ROOT/'prompts/manifests/tokenizer_contract.json')
    for p in ps:
        if digest(p['token_ids'])!=p['token_ids_sha256']:raise ValueError('prompt hash mismatch')
        if p['tokenizer_file_hash']!=contract['tokenizer_file_hash']:raise ValueError('tokenizer mismatch')
    return c,ps


def require_permit(config_path,run_id):
    path=os.environ.get('V41_PERMIT_FILE');nonce=os.environ.get('V41_PERMIT_NONCE')
    if not path or not nonce:raise RuntimeError('GPU_DISABLED: use reviewed-window supervisor')
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT.resolve()/'logs'):raise RuntimeError('permit outside project')
    d=load(p)
    if d['nonce']!=nonce or d['run_id']!=run_id or d['config_sha256']!=file_hash(config_path):raise RuntimeError('invalid permit')
    if time.time()>=d['expires_unix']:raise RuntimeError('permit expired')
    os.kill(d['supervisor_pid'],0)
    return d


class Hooks:
    """Observe CPU request events and dispatch identity; never infer GPU time here."""
    def __init__(self,llm,prompts,torch,profile):
        self.eng=llm.llm_engine
        if type(self.eng.engine_core).__name__!='InprocClient':raise RuntimeError('unverified EngineCore process layout')
        self.core=self.eng.engine_core.engine_core
        self.torch=torch;self.profile=profile;self.prompts=prompts;self.state=None
        self.local=threading.local();self.schedule_ids={};self.next_prompt=0
        self.originals={};self.lock=threading.Lock()

    def mark(self,text):
        if self.profile:self.torch.cuda.nvtx.mark(text)

    @contextlib.contextmanager
    def nvtx(self,text):
        if self.profile:self.torch.cuda.nvtx.range_push(text)
        try:yield
        finally:
            if self.profile:self.torch.cuda.nvtx.range_pop()

    def install(self):
        eng,core,torch=self.eng,self.core,self.torch
        add=eng.add_request;schedule=core.scheduler.schedule
        process=eng.output_processor.process_outputs;execute=core.model_executor.execute_model
        replay=torch.cuda.CUDAGraph.replay;step=eng.step
        signature=inspect.signature(add)
        def add_hook(*args,**kwargs):
            if self.state is None:return add(*args,**kwargs)
            bound=signature.bind_partial(*args,**kwargs).arguments
            before=time.perf_counter_ns()
            result=add(*args,**kwargs)
            # Installed vLLM may rewrite the caller's ID in assign_request_id().
            # add_request() returns the authoritative EngineCore request ID.
            rid=result if isinstance(result,str) else bound.get('request_id')
            if not isinstance(rid,str):raise RuntimeError('unknown engine request-id API')
            if self.next_prompt>=len(self.prompts):raise RuntimeError('unexpected request')
            self.state.enqueue(rid,self.prompts[self.next_prompt],before)
            self.next_prompt+=1
            self.mark('REQUEST_START|'+rid)
            return result
        def schedule_hook(*args,**kwargs):
            if self.state is None:return schedule(*args,**kwargs)
            before={rid:dict(computed=r.num_computed_tokens,prompt_length=r.num_prompt_tokens,
                             ids=list(r.prompt_token_ids or [])) for rid,r in core.scheduler.requests.items()}
            t0=time.perf_counter_ns()
            with self.nvtx('SCHEDULER_ITERATION|CPU_SCHEDULE'):
                s=schedule(*args,**kwargs)
            t1=time.perf_counter_ns()
            for rid in s.num_scheduled_tokens:
                expected=self.state.requests[rid]
                if digest(before[rid]['ids'])!=expected['prompt_hash']:raise RuntimeError('engine input ID mismatch')
            row=self.state.schedule(before,s.num_scheduled_tokens,t0,t1,len(getattr(s,'preempted_req_ids',[])))
            self.schedule_ids[id(s)]=row
            return s
        def execute_hook(*args,**kwargs):
            if self.state is None:return execute(*args,**kwargs)
            s=args[0] if args else kwargs.get('scheduler_output')
            row=self.schedule_ids.get(id(s))
            iteration=row['scheduler_iteration'] if row else 'UNKNOWN'
            phase=row['stage'] if row else 'UNKNOWN'
            self.local.iteration=iteration
            try:
                with self.nvtx(f'ITERATION_DISPATCH|{iteration}|{phase}'):
                    with self.nvtx(phase):return execute(*args,**kwargs)
            finally:self.local.iteration=None
        def replay_hook(graph,*args,**kwargs):
            if self.state is None:return replay(graph,*args,**kwargs)
            iteration=getattr(self.local,'iteration',None)
            with self.lock:
                ix=len(self.state.replays)
                self.state.replays.append(dict(replay_id=ix,scheduler_iteration=iteration,
                                               cpu_start_ns=time.perf_counter_ns(),status='CPU_OBSERVATION'))
            with self.nvtx(f'GRAPH_REPLAY|{ix}|{iteration}'):
                result=replay(graph,*args,**kwargs)
            self.state.replays[ix]['cpu_end_ns']=time.perf_counter_ns()
            if isinstance(iteration,int):self.state.iterations[iteration]['graph_replay_ids'].append(ix)
            return result
        def process_hook(outputs,*args,**kwargs):
            data=[(o.request_id,list(o.new_token_ids)) for o in outputs] if self.state else []
            result=process(outputs,*args,**kwargs)
            now=time.perf_counter_ns()
            for rid,tokens in data:
                if self.state.output(rid,tokens,now):self.mark('FIRST_TOKEN|'+rid)
            return result
        def step_hook(*args,**kwargs):
            result=step(*args,**kwargs)
            if self.state:
                now=time.perf_counter_ns()
                for o in result:
                    if getattr(o,'finished',False):
                        self.state.finish(o.request_id,now);self.mark('REQUEST_END|'+o.request_id)
            return result
        self.originals=dict(add=add,schedule=schedule,execute=execute,process=process,replay=replay,step=step)
        eng.add_request=add_hook;core.scheduler.schedule=schedule_hook
        core.model_executor.execute_model=execute_hook;eng.output_processor.process_outputs=process_hook
        torch.cuda.CUDAGraph.replay=replay_hook
        eng.step=step_hook

    def restore(self):
        o=self.originals
        self.eng.add_request=o['add'];self.core.scheduler.schedule=o['schedule']
        self.core.model_executor.execute_model=o['execute'];self.eng.output_processor.process_outputs=o['process']
        self.torch.cuda.CUDAGraph.replay=o['replay']
        self.eng.step=o['step']


def execute(config_path,run_id,profile,output):
    permit=require_permit(config_path,run_id)  # before torch/vLLM import, always
    out=Path(output).resolve()
    if not out.is_relative_to((ROOT/'raw').resolve()) or out.exists():raise RuntimeError('unsafe or existing output path')
    c,prompts=prepare(config_path)
    project=Path.home()/'fp4-llm'
    model_name='Qwen3-8B-BF16' if c['precision']=='bf16' else 'Qwen3-8B-NVFP4'
    model=project/'models'/model_name
    expected_assets=load(ROOT/'analysis/cpu_environment_audit.json')['tokenizer_asset_hashes'][model_name]
    for name,expected in expected_assets.items():
        if file_hash(model/name)!=expected:raise RuntimeError('model/tokenizer asset changed: '+name)
    import torch
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(model),generation_config=c['generation_config'],**c['engine'])
    hooks=Hooks(llm,prompts,torch,profile!='P0');hooks.install()
    result=dict(schema='v4.1',config=c,run_id=run_id,profile=profile,config_file_sha256=file_hash(config_path),
                prompt_manifest_hash=c['dataset_manifest_sha256'],gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','UNAVAILABLE')),
                environment_versions={p:importlib.metadata.version(p) for p in ['torch','vllm','transformers','flashinfer-python']},
                model_asset_sha256=expected_assets,weight_integrity='UNVERIFIED_FULL_WEIGHT_SHARDS_NOT_HASHED',
                code_sha256={f.name:file_hash(f) for f in Path(__file__).parent.glob('*.py')},
                scope=c['scope'],sampling_excludes_client_network_and_tokenization=True,
                source_model=str(model),async_scheduling=hooks.core.async_scheduling,runs=[])
    ps=[dict(prompt_token_ids=p['token_ids']) for p in prompts]
    trials=c['trials'] if profile=='P0' else 1
    try:
        for _ in range(c['warmup_trials']):
            llm.generate(ps,SamplingParams(max_tokens=c['warmup_output_tokens'],temperature=0.,ignore_eos=True,seed=0),use_tqdm=False)
        torch.cuda.synchronize()  # before full inference; no phase-boundary sync
        for trial in range(trials):
            if profile!='P0':torch.cuda.profiler.start()
            with hooks.nvtx('FULL_INFERENCE'):
                start=time.perf_counter_ns()
                state=TraceState(c['dataset_manifest_sha256'],start)
                hooks.state=state;hooks.next_prompt=0;hooks.schedule_ids.clear()
                outputs=llm.generate(ps,SamplingParams(max_tokens=c['max_output_tokens'],temperature=c['temperature'],
                                      ignore_eos=c['ignore_eos'],seed=c['seed'],stop_token_ids=c['stop_token_ids']),use_tqdm=False)
                end=time.perf_counter_ns()
                for o in outputs:
                    r=state.requests[o.request_id]
                    if r['output_token_ids']!=list(o.outputs[0].token_ids):raise RuntimeError('output instrumentation mismatch')
                    if r['finished_ns'] is None:raise RuntimeError('request completion hook missing')
                hooks.state=None
            torch.cuda.synchronize()  # after CPU response boundary; retained as separate drain
            drain_end=time.perf_counter_ns()
            if profile!='P0':torch.cuda.profiler.stop()
            state.verify_fixed(c['decode_steps'])
            if len(state.requests)!=c['batch']:raise RuntimeError('actual request count mismatch')
            if any(r['stage']=='DECODE' and not r['graph_replay_ids'] for r in state.iterations):
                raise RuntimeError('Decode Graph replay association missing; inspect capture/fallback')
            row=dict(trial=trial,full_cpu_start_ns=start,full_cpu_end_ns=end,post_response_gpu_drain_end_ns=drain_end,
                     full_model_wall_ms=(end-start)/1e6,total_output_tokens=sum(len(r['output_token_ids']) for r in state.requests.values()),
                     requests=state.requests,request_summary=state.summaries(),scheduler_iterations=state.iterations,
                     graph_replays=state.replays,validation='RUNTIME_ASSERTIONS_PASSED; GPU causal mapping still requires trace')
            result['runs'].append(row)
            save(output,result,overwrite=trial>0)
            print('V41_TRIAL',run_id,trial,row['full_model_wall_ms'],'ms',flush=True)
    finally:
        hooks.restore();hooks.eng.engine_core.shutdown()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--run-id',default='cpu_plan')
    ap.add_argument('--profile',choices=['P0','P1','P2'],default='P0');ap.add_argument('--output');ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();c,ps=prepare(a.config)
    if not a.execute:
        print(dict(status='CPU_PLAN_ONLY',experiment=c['experiment'],precision=c['precision'],batch=len(ps),
                   prompt_lengths=[p['actual_token_length'] for p in ps],max_output_tokens=c['max_output_tokens'],profile=a.profile))
    else:
        if not a.output:raise ValueError('output required')
        execute(a.config,a.run_id,a.profile,a.output)
