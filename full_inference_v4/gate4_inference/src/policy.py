"""Disabled-by-default, evidence-specific admission; not a proven scheduler."""
from dataclasses import dataclass
import math
from useful_work import UsefulWork

@dataclass
class Decision:
    execute_auxiliary:bool
    reason:str
    evidence_id:str|None=None
    status:str='CPU_DESIGN_NOT_GPU_VALIDATED'

def decide(work:UsefulWork,current,evidence,*,enabled=False,loss_limit=.05,p95_limit=.05):
    if not enabled:return Decision(False,'policy disabled: original inference path')
    try:work.admission()
    except ValueError as error:return Decision(False,str(error))
    # Resource similarity to a probe is not sufficient. Calibration must name
    # this *actual useful kernel*, its shape, phase and code/config version.
    keys=['device_uuid','precision','phase','auxiliary_kernel_hash','auxiliary_config_hash',
          'primary_family','workload_manifest_hash']
    matched=[e for e in evidence if all(e.get(k)==current.get(k) and current.get(k) is not None for k in keys)]
    matched=[e for e in matched if e.get('actual_useful_consumer_verified') and e.get('paired_blocks',0)>=5
             and e.get('independent_processes',0)>=2 and e.get('timeline_overlap_verified')
             and e.get('p0_formal') and e.get('same_work_verified')]
    if not matched:return Decision(False,'no candidate-specific validated Gate-4 evidence')
    for e in matched:
        needed=['primary_interference_ci95_high','p95_loss','joint_speedup_ci95_low',
                'probe_throughput_retention','resource_aware_vs_ordinary_ci95_low','auxiliary_duration_p95_ns']
        if any(not isinstance(e.get(k),(int,float)) or not math.isfinite(e[k]) for k in needed):continue
        if not isinstance(current.get('now_ns'),int) or not isinstance(current.get('auxiliary_deadline_ns'),int):continue
        if e['primary_interference_ci95_high']>loss_limit or e['p95_loss']>p95_limit:continue
        if e['joint_speedup_ci95_low']<=1 or e['probe_throughput_retention']<=0:continue
        if e.get('ordinary_concurrency_comparison_status')!='VERIFIED':continue
        if e['resource_aware_vs_ordinary_ci95_low']<=1:continue
        if current['now_ns']+e['auxiliary_duration_p95_ns']>current['auxiliary_deadline_ns']:continue
        return Decision(True,'matched calibrated useful task within budget',e['evidence_id'])
    return Decision(False,'calibration exists but delay/deadline/benefit limits are not met')
