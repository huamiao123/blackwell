"""Paired metrics with exact work, scope and profiling boundaries."""
from statistics import mean,stdev
import math
from contracts import ContractError,validate_work_equal

def interval_union(xs):
    out=[]
    for a,b in sorted(xs):
        if b<a:raise ContractError('negative interval')
        if b==a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out

def overlap(a,b):
    if a[1]<=a[0] or b[1]<=b[0]:raise ContractError('zero/negative kernel duration')
    n=max(0,min(a[1],b[1])-max(a[0],b[0]))
    return n,n/min(a[1]-a[0],b[1]-b[0])

def paired(block,order):
    required=['a_alone','b_alone','serial_'+order,'parallel_'+order]
    rows=[block[m] for m in required]
    if any(r['profile']!='P0' for r in rows):raise ContractError('P2 timing is not a P0 baseline')
    validate_work_equal(rows)
    units=[(r.get('a_completed_units',0),r.get('b_completed_units',0)) for r in rows]
    au,bu=units[0][0],units[1][1]
    if au<=0 or bu<=0 or units!=[(au,0),(0,bu),(au,bu),(au,bu)]:
        raise ContractError('actual completed A/B work differs between conditions')
    if len({r['measurement_scope'] for r in rows})!=1:raise ContractError('mixed timing scopes')
    a,b,serial,parallel=rows
    if not serial.get('serial_measured'):raise ContractError('serial must be an actual run')
    if min(a['a_ns'],b['b_ns'],parallel['a_ns'],parallel['b_ns'],serial['joint_ns'],parallel['joint_ns'])<=0:
        raise ContractError('invalid duration')
    return dict(primary_interference=parallel['a_ns']/a['a_ns']-1,
        probe_throughput_retention=b['b_ns']/parallel['b_ns'],
        joint_speedup=serial['joint_ns']/parallel['joint_ns'],
        primary_ns=parallel['a_ns'],probe_ns=parallel['b_ns'],serial_ns=serial['joint_ns'],joint_ns=parallel['joint_ns'],
        order=order,measurement_scope=parallel['measurement_scope'])

def confidence(values):
    if len(values)<5:raise ContractError('at least five valid paired blocks')
    n=len(values);m=mean(values);s=stdev(values)
    # Initial design has n=5. Do not pretend an arbitrary normal interval is t.
    t={4:2.776445105,5:2.570581836,6:2.446911851,7:2.364624252,8:2.306004135,9:2.262157163}
    if n-1 not in t:raise ContractError('unsupported t degrees of freedom; explicitly extend statistics method')
    half=t[n-1]*s/math.sqrt(n)
    return dict(n_paired_blocks=n,mean=m,sd=s,ci95_low=m-half,ci95_high=m+half,
                assumption='conditional block independence within process; not a cross-device population CI')

def completed_within_primary(a,b,total_work):
    if b[0]>=a[0] and b[1]<=a[1]:return total_work
    return 'UNVERIFIED_NO_SUBKERNEL_PROGRESS_TELEMETRY'
