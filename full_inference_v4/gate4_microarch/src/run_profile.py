"""One foreground NCU diagnostic using the shared Gate 2 supervisor."""
import argparse
import datetime as dt
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
sys.path.insert(0, str(V4/'src'))
import run_supervised as supervisor
from common import load

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--operator', choices=('qkv_proj','gate_up_proj','o_proj','down_proj'), required=True)
    parser.add_argument('--counters', choices=('pilot','resources','stalls','cache_read','pc'), required=True)
    parser.add_argument('--cache', choices=('none','all'), default='none')
    args, rest = parser.parse_known_args()
    counters = load(HERE.parent/'configs/counters.json')[args.counters]

    def window(w,c,now,host):
        target = [c['precision'], args.operator, args.counters, args.cache]
        if not (w.get('enabled') and w.get('gate')=='microarch' and w.get('hostname')==host
                and w.get('allowed')==target and c['experiment']=='E2' and w.get('authorization_reference')):
            raise ValueError('invalid microarchitecture authorization window')
        start,end = (dt.datetime.fromisoformat(w[k]) for k in ('not_before','expires_at'))
        if not (start.tzinfo and end.tzinfo and start.timestamp() <= now < end.timestamp()
                and 0 < (end-start).total_seconds() <= 300):
            raise ValueError('invalid window time')
        return end.timestamp()

    def command(worker,profile,stem,env):
        if profile != 'P3':
            raise ValueError('requires P3')
        reports = HERE.parent/'reports'
        reports.mkdir(exist_ok=True)
        report = reports/stem.name
        if report.with_suffix('.ncu-rep').exists():
            raise FileExistsError(report)
        return ['sudo','-n','/usr/local/sbin/gpu-profile','ncu','--target-processes','all',
                '--profile-from-start','off','--graph-profiling','node','--launch-count','1',
                '--replay-mode','kernel','--clock-control','none','--cache-control',args.cache,
                '--metrics',','.join(counters),'--export',str(report),'env',
                *[f'{k}={v}' for k,v in env.items()],worker[0],str(HERE/'profile_projection.py'),
                '--config',worker[worker.index('--config')+1],'--run-id',stem.name,'--operator',args.operator]

    supervisor.validate_window = window
    supervisor.profiler_command = command
    sys.argv = [sys.argv[0],*rest]
    supervisor.main(profile_choices=('P3',))

if __name__=='__main__':
    main()
