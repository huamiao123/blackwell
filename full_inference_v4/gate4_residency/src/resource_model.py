"""Necessary resource bounds, not an achieved-occupancy or mixed-kernel oracle."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate4_spatial/src'))
from g4_common import *

def footprint(cta,caps,reg_quantum=None,shared_quantum=None):
    warps=math.ceil(cta['threads']/caps['warp_size'])
    raw_regs=cta['threads']*cta['registers_per_thread']
    # A supplied register quantum is per warp, and must carry separate audited
    # architecture provenance. No default SM120 granularity is invented.
    regs=warps*math.ceil(cta['registers_per_thread']*caps['warp_size']/reg_quantum)*reg_quantum if reg_quantum else raw_regs
    shared=cta['static_shared_bytes']+cta['dynamic_shared_bytes']
    if shared_quantum:shared=math.ceil(shared/shared_quantum)*shared_quantum
    return dict(registers=regs,raw_registers=raw_regs,shared=shared,warps=warps,threads=cta['threads'],ctas=1)

def homogeneous(cta,caps):
    f=footprint(cta,caps)
    ratios=dict(registers=caps['registers_per_sm']//f['registers'] if f['registers'] else caps['ctas_per_sm'],
                shared=caps['shared_per_sm']//f['shared'] if f['shared'] else caps['ctas_per_sm'],
                warps=caps['warps_per_sm']//f['warps'],threads=caps['threads_per_sm']//f['threads'],ctas=caps['ctas_per_sm'])
    return dict(raw_homogeneous_cta_upper_bound=min(ratios.values()),**{k+'_upper_bound':v for k,v in ratios.items()},
                occupancy_api='NOT_RUN_CPU_ONLY',actual_occupancy='UNAVAILABLE',uncertainty='allocation rounding and function attributes not fully known')

def mixed(a,b,caps,n_a=1,n_b=1,reg_quantum=None,shared_quantum=None):
    if min(n_a,n_b)<1:raise ValueError('positive CTA multiplicities required')
    fa=footprint(a,caps,reg_quantum,shared_quantum);fb=footprint(b,caps,reg_quantum,shared_quantum)
    totals={k:n_a*fa[k]+n_b*fb[k] for k in ['registers','shared','warps','threads','ctas']}
    limits=dict(registers=caps['registers_per_sm'],shared=caps['shared_per_sm'],warps=caps['warps_per_sm'],
                threads=caps['threads_per_sm'],ctas=caps['ctas_per_sm'])
    flags={k+'_limited':totals[k]>limits[k] for k in limits}
    per_block_invalid=any(x['threads']>caps['max_threads_per_block'] or x['registers']>caps['registers_per_block']
        or x['shared']>caps['shared_optin_per_block'] for x in [fa,fb])
    possible=not any(flags.values()) and not per_block_invalid
    return dict(same_sm_theoretical_feasibility='NECESSARY_BOUNDS_PASS_NOT_SUFFICIENT' if possible else 'INFEASIBLE_BY_NECESSARY_BOUND',
        main_ctas_assumed=n_a,probe_ctas_assumed=n_b,**flags,per_block_invalid=per_block_invalid,
        registers_total=totals['registers'],shared_total=totals['shared'],warps_total=totals['warps'],threads_total=totals['threads'],
        allocation_granularity='AUDITED_MODEL_INPUT' if reg_quantum and shared_quantum else 'UNAVAILABLE_NOT_ASSUMED',
        uncertainty='homogeneous occupancy APIs do not directly predict mixed kernels; partition/cluster/cache-config constraints remain',
        actual_kernel_overlap='UNAVAILABLE_NO_GPU_RUN',same_sm_evidence='SAME_SM_CO_RESIDENCY_UNVERIFIED')

def occupancy_api_query(driver,func,threads,dynamic_shared):
    """Deferred authorized-context query; no CUDA import or query at import time."""
    def get(result):
        if int(result[0]):raise RuntimeError(str(result[0]))
        return result[1]
    attrs={name:get(driver.cuFuncGetAttribute(getattr(driver.CUfunction_attribute,name),func)) for name in
           ['CU_FUNC_ATTRIBUTE_NUM_REGS','CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES','CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
            'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES']}
    blocks=get(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(func,threads,dynamic_shared))
    return dict(api='cuOccupancyMaxActiveBlocksPerMultiprocessor',threads=threads,dynamic_shared=dynamic_shared,
                blocks_per_sm=blocks,attributes=attrs,scope='one function only; not mixed CTA co-residency',actual_occupancy='UNAVAILABLE')

def main():
    caps=load(SPATIAL/'configs/hardware_from_gate3.json');rows=[]
    for c in read_csv(V4/'gate4_residency/analysis/cta_resource_table.csv'):
        c={k:(int(v) if k in ['threads','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','grid_ctas'] else v) for k,v in c.items()}
        rows.append(dict(precision=c['precision'],logical_op=c['logical_op'],**homogeneous(c,caps)))
    write_csv(V4/'gate4_residency/analysis/theoretical_residency.csv',rows)
    print('Derived CPU necessary upper bounds from Gate-3 data; no Occupancy API called')

if __name__=='__main__':main()
