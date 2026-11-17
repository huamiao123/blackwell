"""Pure command builder. It never invokes nsys or starts a model itself."""
from pathlib import Path


def profiler_command(worker,profile,stem,env):
    if profile=='P0':return worker
    if profile not in ['P1','P2']:raise ValueError('P3 requires a separately reviewed kernel-specific plan')
    flags=['--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
           '--cuda-graph-trace='+('graph' if profile=='P1' else 'node'),
           '--capture-range=cudaProfilerApi','--capture-range-end=stop-shutdown',
           '--kill=none','--output='+str(stem)]
    if profile=='P1':flags+=['--gpu-metrics-devices=0','--gpu-metrics-set=gb20x','--gpu-metrics-frequency=10000']
    return ['sudo','-n','--preserve-env=TMPDIR','/usr/local/sbin/gpu-profile','nsys','profile',*flags,
            'env',*[f'{k}={v}' for k,v in env.items()],*worker]


def export_command(rep,dest):
    return ['nsys','export','--type=sqlite','--output='+str(dest),str(rep)]
