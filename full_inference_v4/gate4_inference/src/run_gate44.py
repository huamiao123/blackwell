"""Run one bounded Gate 4.4 job through the frozen E2 supervisor."""

import argparse
import datetime as dt
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / 'src'))
import run_supervised as supervisor


def validate_window(window, config, now, hostname, phase, condition, profile):
    if not window.get('enabled') or window.get('gate') != '4.4':
        raise ValueError('Gate 4.4 window missing')
    if window['hostname'] != hostname or not window.get('authorization_reference'):
        raise ValueError('wrong host or missing authorization')
    expected = dict(experiment='E2', precision=config['precision'], phase=phase,
                    condition=condition, profile=profile)
    if config['experiment'] != 'E2' or window['allowed_workload'] != expected:
        raise ValueError('workload does not match this reviewed E2 window')
    start, end = (dt.datetime.fromisoformat(window[key]) for key in ('not_before', 'expires_at'))
    if (start.tzinfo is None or end.tzinfo is None or not start.timestamp() <= now < end.timestamp()
            or (end - start).total_seconds() > 900):
        raise ValueError('invalid or expired bounded window')
    return end.timestamp()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', choices=('decode', 'prefill', 'lifecycle'), required=True)
    parser.add_argument('--condition', choices=('parallel', 'serial', 'empty', 'a_alone', 'paired'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', choices=('P0', 'P1', 'P2'), required=True)
    args, remaining = parser.parse_known_args()
    if not args.run_id.startswith('gate44_') or (args.condition == 'paired' and args.profile != 'P0'):
        raise ValueError('invalid Gate 4.4 run prefix or profile')
    builder = supervisor.profiler_command

    def command(worker, profile, stem, env):
        worker = [worker[0], str(HERE.with_name('gate44_worker.py')), *worker[2:],
                  '--phase', args.phase, '--condition', args.condition]
        return builder(worker, profile, stem, env)

    supervisor.validate_window = lambda w, c, now, host: validate_window(
        w, c, now, host, args.phase, args.condition, args.profile)
    supervisor.profiler_command = command
    sys.argv = [sys.argv[0], '--run-id', args.run_id, '--profile', args.profile, *remaining]
    supervisor.main()


if __name__ == '__main__':
    main()
