"""One existing shared-GPU supervisor invocation for E2 node placement."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate4_inference/src'))
from run_gate44 import validate_window,supervisor
builder=supervisor.profiler_command

def cmd(worker,profile,stem,env):
    if not stem.name.startswith('nodetest_') or profile not in ('P2','P0'):raise ValueError('invalid target run')
    args=builder([worker[0],str(Path(__file__).with_name('target_worker.py')),*worker[2:]],profile,stem,env)
    if profile=='P2':
        args[args.index('--capture-range-end=stop-shutdown')]='--capture-range-end=stop'
    return args

if __name__=='__main__':
    profile=sys.argv[sys.argv.index('--profile')+1]
    supervisor.validate_window=lambda w,c,t,h: validate_window(w,c,t,h,'decode_nodes','paired',profile)
    supervisor.profiler_command=cmd
    supervisor.main()
