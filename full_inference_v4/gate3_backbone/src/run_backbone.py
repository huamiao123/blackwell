"""Finite supervised Gate-3 continuation. Existing GPU safety checks retained."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
sys.path.insert(0, str(V4 / 'gate3_attribution/src'))
import run_supervised as supervisor
from run_diagnostic import validate_window


def main():
    if '--profile' not in sys.argv or sys.argv[sys.argv.index('--profile')+1] != 'P2':
        raise ValueError('P2 only')
    if '--run-id' not in sys.argv or not sys.argv[sys.argv.index('--run-id')+1].startswith('gate3_bb_'):
        raise ValueError('new backbone run ID required')
    build = supervisor.profiler_command

    def command(worker, profile, stem, env):
        worker = list(worker)
        worker[1] = str(HERE.with_name('backbone_worker.py'))
        return build(worker, profile, stem, env)

    supervisor.validate_window = validate_window
    supervisor.profiler_command = command
    supervisor.main()


if __name__ == '__main__':
    main()
