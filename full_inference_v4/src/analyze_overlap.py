"""CPU-only same-work metrics and physical interval intersection."""
from common import paired_metrics,intersection,duration


def physical_overlap(a,b,a_end):
    inside=[(s,min(e,a_end)) for s,e in b if s<a_end]
    return dict(kernel_intersection_ns=duration(intersection(a,inside)),
                fully_completed_b_kernels=sum(e<=a_end for s,e in b),
                partial_b_kernels=sum(s<a_end<e for s,e in b),
                same_sm_residency='UNVERIFIED',
                interpretation='Time overlap and whole-kernel completion only; partial work not prorated')
