"""Single, time-bounded job using the shared RTX 5090 ownership guard."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate4_inference/src'))
from run_gate44 import validate_window,supervisor
builder=supervisor.profiler_command

def command(worker,profile,stem,env):
    if not stem.name.startswith('qkready_') or profile not in ('P0','P2'):
        raise ValueError('outside Q/K pilot')
    cmd=builder([worker[0],str(Path(__file__).with_name('worker.py')),*worker[2:]],profile,stem,env)
    if profile=='P2':
        cmd[cmd.index('--capture-range-end=stop-shutdown')]='--capture-range-end=stop'
    return cmd

if __name__=='__main__':
    profile=sys.argv[sys.argv.index('--profile')+1]
    supervisor.validate_window=lambda w,c,t,h:validate_window(w,c,t,h,'decode_qk','paired',profile)
    supervisor.profiler_command=command
    supervisor.main()
