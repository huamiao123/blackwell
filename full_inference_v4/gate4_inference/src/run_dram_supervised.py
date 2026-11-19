"""Run one streaming-probe measurement via the existing shared GPU supervisor."""
import argparse
from pathlib import Path
import sys

from run_gate44 import supervisor, validate_window

HERE = Path(__file__).resolve()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', choices=('dram_standalone', 'dram_decode'), required=True)
    parser.add_argument('--condition', choices=('acceptance', 'paired', 'serial', 'parallel'), required=True)
    parser.add_argument('--profile', choices=('P0', 'P1', 'P2'), required=True)
    parser.add_argument('--run-id', required=True)
    args, rest = parser.parse_known_args()
    standalone = args.phase == 'dram_standalone'
    allowed = ({('acceptance','P0'), ('acceptance','P1')} if standalone else
               {('paired','P0'), ('parallel','P2'), ('serial','P2'), ('parallel','P1'), ('serial','P1')})
    if not args.run_id.startswith('dram44_') or (args.condition,args.profile) not in allowed:
        raise ValueError('unexpected DRAM measurement')
    builder = supervisor.profiler_command

    def command(worker, profile, stem, env):
        script = 'dram_probe_gpu.py' if standalone else 'dram_decode_worker.py'
        return builder([worker[0], str(HERE.with_name(script)), *worker[2:],
                        '--condition', args.condition], profile, stem, env)

    supervisor.validate_window = lambda w,c,t,h: validate_window(w,c,t,h,args.phase,args.condition,args.profile)
    supervisor.profiler_command = command
    sys.argv = [sys.argv[0], '--profile', args.profile, '--run-id', args.run_id, *rest]
    supervisor.main()


if __name__ == '__main__':
    main()
