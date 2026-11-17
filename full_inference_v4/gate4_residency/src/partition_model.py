"""Conservative register-placement search under an explicitly sourced CPU model."""
import functools,math

def placements(total,bins,limit):
    if bins==1:
        if 0<=total<=limit:yield (total,)
        return
    for x in range(min(total,limit)+1):
        for rest in placements(total-x,bins-1,limit):yield (x,)+rest

def register_partition_fit(a,b,caps,granularity):
    partitions=granularity['subpartitions'];capacity=caps['registers_per_sm']//partitions
    q=granularity['register_quantum_per_warp']
    ra=math.ceil(a['registers_per_thread']*caps['warp_size']/q)*q
    rb=math.ceil(b['registers_per_thread']*caps['warp_size']/q)*q
    wa=math.ceil(a['threads']/caps['warp_size']);wb=math.ceil(b['threads']/caps['warp_size'])
    for aa in placements(wa,partitions,capacity//ra):
        remaining=[capacity-n*ra for n in aa]
        if sum(r//rb for r in remaining)>=wb:
            return dict(status='REGISTER_PARTITION_MODEL_HAS_PLACEMENT',main_warps_per_partition=aa,
                        uncertainty='placement is not guaranteed by the scheduler; other constraints not included')
    return dict(status='REGISTER_PARTITION_MODEL_NO_PLACEMENT',main_warp_allocation=ra,probe_warp_allocation=rb,
                capacity_per_partition=capacity,uncertainty='CPU CUDA-header model, not actual mixed-kernel occupancy or dynamic register telemetry')
