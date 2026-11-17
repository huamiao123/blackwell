"""Evidence levels never turn a resource fit or overlap into direct same-SM proof."""
def classify(theory,overlap_record=None,cta_trace=None):
    out=dict(theory=theory['same_sm_theoretical_feasibility'],
             concurrency='UNVERIFIED_NO_GPU_TIMELINE',same_sm='SAME_SM_CO_RESIDENCY_UNVERIFIED',
             spatial_interpretation='UNVERIFIED')
    if overlap_record and overlap_record.get('actual_gpu_intervals_verified') and overlap_record.get('overlap_ns',0)>0:
        out['concurrency']='CONCURRENT_KERNEL_EXECUTION_VERIFIED'
        if theory['same_sm_theoretical_feasibility']=='INFEASIBLE_BY_NECESSARY_BOUND':
            out['spatial_interpretation']='PARTIALLY_SUPPORTED_DIFFERENT_SM_OR_CTA_INTERLEAVING'
    # No primary CTA-location collection exists for these original Gate-3
    # kernels. Probe-only SM IDs or an instrumented replacement are insufficient.
    if cta_trace:
        out['direct_telemetry_status']='NO_VALIDATED_PRIMARY_CTA_COLLECTOR_IMPLEMENTED'
        # An untrusted dictionary of asserted booleans is not a hardware trace.
        # Add a separately reviewed, hash-bound primary CTA collector before
        # enabling a direct same-SM verdict. Probe-only %smid cannot enable it.
    return out
