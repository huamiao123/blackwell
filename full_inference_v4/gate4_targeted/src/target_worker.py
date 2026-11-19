"""Run the original E2 requests, choosing a diagnostic graph at known Decode steps."""
import argparse
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve()
V4=HERE.parents[2]
sys.path[:0]=[str(V4/'src'),str(V4/'gate4_inference/src'),str(V4/'gate3_backbone/src')]
import run_full_inference as original
from gate44_worker import Gate44Hooks,SPATIAL
from graph_binding import checked
from target_graph import TargetGraphs
from protocol import MODES,STEPS,schedules

class TargetHooks(Gate44Hooks):
    def install(self):
        from cuda.bindings import driver,runtime
        self.driver=driver
        self.graphs.hook.remove()
        self.targets=STEPS
        self.conditions=('probe_buffers',)
        self.trial=0
        self.logs=[]
        self.measured=[]
        self.native=self.torch.cuda.CUDAGraph.replay
        self.model=self.core.model_executor.driver_worker.worker.model_runner.model
        output=self.prepare_probe()
        self.warm_buffer=output
        self.warm_launches=0
        self.torch.cuda.synchronize()
        self.baseline=self.validate_alone(output)
        self.ctx=int(checked(driver.cuCtxGetCurrent()))
        def diagnostic(graph,*args,**kwargs):
            if self.state is None:
                if not self.graphs.variants:
                    self.graphs.build(graph,self,self.model,self.precision,SPATIAL,MODES)
                if self.warm_variants and graph is self.graphs.selected:
                    warm_mode=MODES[self.warm_launches%len(MODES)]
                    self.warm_launches+=1
                    if warm_mode!='original':
                        return self.graphs.launch(warm_mode,len(STEPS),self.torch.cuda.current_stream(),runtime)
                return self.native(graph,*args,**kwargs)
            step=getattr(self.local,'iteration',None)
            if graph is not self.graphs.selected:raise ValueError('unexpected E2 decode graph')
            if self.warm_variants and self.warm_launches<2*len(MODES):raise ValueError('incomplete diagnostic warmup')
            if step not in STEPS:return self.native(graph,*args,**kwargs)
            if (self.state.iterations[step]['stage'],self.state.iterations[step]['active_request_count'])!=('DECODE',8):
                raise ValueError('target shape/stage changed')
            slot=STEPS.index(step)
            mode=self.plan[self.trial][slot]
            if int(checked(driver.cuCtxGetCurrent()))!=self.ctx:raise RuntimeError('CUDA context changed')
            with self.nvtx(f'TARGET_NODE|{step}|{mode}'):
                if mode=='original':result=self.native(graph,*args,**kwargs)
                else:result=self.graphs.launch(mode,slot,self.torch.cuda.current_stream(),runtime)
            self.logs.append(dict(iteration=step,mode=mode))
            return result
        self.torch.cuda.CUDAGraph.replay=diagnostic
        original.HooksBase.install(self)
        TargetHooks.active=self

    def finish(self):
        modes=self.plan[self.trial]
        if [(r['iteration'],r['mode']) for r in self.logs]!=list(zip(STEPS,modes)):raise ValueError('missing target step')
        for slot,(mode,row) in enumerate(zip(modes,self.logs)):
            output=self.buffers[0,slot]
            if mode in ('original','sham'):
                if not self.torch.isnan(output).all().item():raise ValueError('unexpected probe output')
            else:row.update(self.verify_output(output))
        self.measured.append(self.logs)
        self.logs=[]
        self.trial+=1
        self.buffers.fill_(float('nan'))
        self.torch.cuda.synchronize()

    def restore(self):
        original.HooksBase.restore(self)
        self.torch.cuda.CUDAGraph.replay=self.native
        self.torch.cuda.synchronize()
        self.graphs.close()

def main():
    p=argparse.ArgumentParser()
    for arg in ('config','run-id','profile','output'):p.add_argument('--'+arg,required=True)
    p.add_argument('--execute',action='store_true',required=True)
    a=p.parse_args()
    original.require_permit(a.config,a.run_id)
    import torch
    from cuda.bindings import driver
    TargetHooks.graphs=TargetGraphs(torch,driver)
    TargetHooks.warm_variants=a.profile=='P0'
    TargetHooks.plan=schedules(a.profile,71011 if '_r2_' not in a.run_id else 84239)
    prepare,save=original.prepare,original.save
    def configure(path):
        config,ps=prepare(path)
        if (config['experiment'],config['batch'],config['decode_steps'])!=('E2',8,256):raise ValueError('wrong workload')
        TargetHooks.precision=config['precision']
        config['trials']=len(TargetHooks.plan)
        return config,ps
    def record(path,data,**kw):
        if data.get('schema')=='v4.1':
            h=TargetHooks.active
            if len(data['runs'])>len(h.measured):h.finish()
            data['target_nodes']=dict(variants=h.graphs.info,work=h.work,probe_sha256=h.cubin_sha,
                                      baseline=h.baseline,trials=h.measured,layer=18,
                                      native_signature=h.graphs.native_signature,
                                      warm_variant_launches=h.warm_launches)
        save(path,data,**kw)
    original.HooksBase=original.Hooks
    original.Hooks=TargetHooks
    original.prepare,original.save=configure,record
    original.execute(a.config,a.run_id,a.profile,a.output)
if __name__=='__main__':
    main()
