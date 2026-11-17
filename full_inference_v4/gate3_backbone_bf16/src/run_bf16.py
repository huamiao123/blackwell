"""Finite supervised Gate-3 continuation. Existing GPU safety checks retained."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
sys.path.insert(0, str(V4 / 'gate3_attribution/src'))
import run_supervised as supervisor
from run_diagnostic import validate_window as nvfp4_window


def validate_window(w,c,now,hostname):
    # Same window checks as the reviewed parent, with an explicit BF16 scope.
    # No BF16 workload is presented to the NVFP4-only validator as an NVFP4 run.
    import datetime as dt
    if not w.get('enabled') or w.get('gate')!=3 or w.get('reviewed_gates')!=[0,1,2]:
        raise ValueError('Gate-3 window/review missing')
    if c['experiment']!='E2' or c['precision']!='bf16' or w['allowed_configs']!=['E2_bf16']:
        raise ValueError('BF16 control scope mismatch')
    if hostname!=w['hostname'] or not w.get('authorization_reference'):
        raise ValueError('host/authorization mismatch')
    start,end=(dt.datetime.fromisoformat(w[k]) for k in ['not_before','expires_at'])
    if start.tzinfo is None or end.tzinfo is None or not start.timestamp()<=now<end.timestamp():
        raise ValueError('outside bounded window')
    if end.timestamp()-start.timestamp()>1200:raise ValueError('window too long')
    return end.timestamp()


def main():
    if '--profile' not in sys.argv or sys.argv[sys.argv.index('--profile')+1] != 'P2':
        raise ValueError('P2 only')
    if '--run-id' not in sys.argv or not sys.argv[sys.argv.index('--run-id')+1].startswith('gate3_bb_'):
        raise ValueError('new backbone run ID required')
    build = supervisor.profiler_command

    def command(worker, profile, stem, env):
        worker = list(worker)
        worker[1] = str(HERE.with_name('bf16_worker.py'))
        return build(worker, profile, stem, env)

    supervisor.validate_window = validate_window
    supervisor.profiler_command = command
    supervisor.main()


if __name__ == '__main__':
    main()
