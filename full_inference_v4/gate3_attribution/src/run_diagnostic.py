"""Gate-3 extension of the reviewed supervisor; every safety check is retained."""
import datetime as dt
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
import run_supervised as supervisor


def validate_window(w, c, now, hostname):
    if not w.get('enabled'):
        raise ValueError('GPU_WINDOW_DISABLED')
    if w.get('gate') != 3 or w.get('reviewed_gates') != [0, 1, 2]:
        raise ValueError('Gate-3 review missing')
    if c['experiment'] != 'E2' or c['precision'] != 'nvfp4' or w['allowed_configs'] != ['E2_nvfp4']:
        raise ValueError('single diagnostic scope mismatch')
    if hostname != w['hostname'] or not w.get('authorization_reference'):
        raise ValueError('host/authorization mismatch')
    start, end = (dt.datetime.fromisoformat(w[k]) for k in ['not_before', 'expires_at'])
    if start.tzinfo is None or end.tzinfo is None or not start.timestamp() <= now < end.timestamp():
        raise ValueError('outside bounded authorization window')
    if end.timestamp() - start.timestamp() > 1200:
        raise ValueError('diagnostic window too long')
    return end.timestamp()


def main():
    if '--profile' not in sys.argv or sys.argv[sys.argv.index('--profile') + 1] != 'P2':
        raise ValueError('this reviewed diagnostic allows P2 only')
    if '--run-id' not in sys.argv or not sys.argv[sys.argv.index('--run-id') + 1].startswith('gate3_'):
        raise ValueError('new Gate-3 run prefix required')
    original_builder = supervisor.profiler_command

    def command(worker, profile, stem, env):
        assert profile == 'P2'
        worker = list(worker)
        worker[1] = str(HERE.with_name('diagnostic_worker.py'))
        return original_builder(worker, profile, stem, env)

    supervisor.validate_window = validate_window
    supervisor.profiler_command = command
    # Original state checks, UUID validation, nonce/permit, per-run timeout,
    # process ownership monitor, conflict stop, cleanup and export remain intact.
    supervisor.main()


if __name__ == '__main__':
    main()
