"""Run the original full E2 with identical model kernels in three dependency configurations."""
import argparse
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'gate3_backbone/src')]
import run_full_inference as original
from graph_binding import checked
from branches import Branches

MODES = ('original','sham','fork')

def schedule(seed, blocks):
    rng = random.Random(seed)
    return [m for _ in range(blocks) for m in rng.sample(MODES,len(MODES))]

def main():
    p = argparse.ArgumentParser()
    for opt in ('config','run-id','profile','output'):
        p.add_argument('--'+opt,required=True)
    p.add_argument('--execute',required=True,action='store_true')
    a = p.parse_args()
    original.require_permit(a.config,a.run_id)
    import torch
    from cuda.bindings import driver as d, runtime
    obs = Branches(torch,d)
    seed = 2743 if '_r1_' in a.run_id else 6451
    modes = schedule(seed,5)
    step_modes = schedule(seed,86)[:256]
    meta = dict(profile=a.profile,seed=seed,step_modes=step_modes,p0_modes=modes)
    prepare,save = original.prepare,original.save
    def configure(path):
        c,ps=prepare(path)
        if (c['experiment'],c['batch'],c['decode_steps'])!=('E2',8,256):
            raise ValueError('expected E2')
        c['trials']=len(modes) if a.profile=='P0' else 1
        return c,ps
    class Hooks(original.Hooks):
        def install(self):
            native=torch.cuda.CUDAGraph.replay
            self.native=native
            self.warm=0
            self.trial=0
            def replay(graph,*args,**kwargs):
                if not obs.executables:
                    model=self.core.model_executor.driver_worker.worker.model_runner.model
                    obs.build(graph,model)
                    meta.update(layers=obs.records,kernels=obs.kernel_count,execs={'original':obs.native_exec,**{
                        m:int(checked(d.cuGraphExecGetId(x[1]))) for m,x in obs.executables.items()}})
                if graph is not obs.graph:
                    raise RuntimeError('unexpected replay graph')
                if self.state is None:
                    mode=MODES[self.warm%len(MODES)]
                    self.warm+=1
                    meta['warm_replays']=self.warm
                else:
                    step=self.local.iteration
                    if not 1<=step<=256 or self.state.iterations[step]['active_request_count']!=8:
                        raise ValueError('unexpected replay or actual batch')
                    mode=step_modes[step-1] if a.profile=='P2' else modes[self.trial]
                if mode=='original':
                    return native(graph,*args,**kwargs)
                checked(runtime.cudaGraphLaunch(int(obs.executables[mode][1]),torch.cuda.current_stream().cuda_stream))
            torch.cuda.CUDAGraph.replay=replay
            super().install()
            meta['hook']=self
        def restore(self):
            super().restore()
            torch.cuda.CUDAGraph.replay=self.native
            torch.cuda.synchronize()
            obs.close()
    def record(path,data,**kw):
        if 'runs' in data and 'hook' in meta:
            meta['hook'].trial=len(data['runs'])
            data['branches']={k:v for k,v in meta.items() if k!='hook'}
        save(path,data,**kw)
    original.Hooks=Hooks
    original.save=record
    original.prepare=configure
    original.execute(a.config,a.run_id,a.profile,a.output)

if __name__=='__main__':
    main()
