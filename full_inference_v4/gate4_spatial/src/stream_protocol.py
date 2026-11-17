"""Same-context, explicit-stream queue protocol with an optional device-memory gate.

Backend methods are an injectable CUDA adapter. CPU tests use a recorder;
no CUDA package is imported here. Do not substitute an unrecorded event gate.
"""
from contracts import ContractError
from plan import MODES

def submit(backend,mode,a,b):
    if mode not in MODES:raise ContractError('unknown condition')
    if backend.context(a)!=backend.context(b):raise ContractError('different CUDA contexts')
    if not backend.stream_memory_ops_supported():raise ContractError('device gate unsupported; no silent spin/sleep fallback')
    backend.reset_gate_and_finish_initialization()  # outside measured interval
    sa,sb,ctrl=backend.nonblocking_streams()
    backend.wait_value(sa,1);backend.wait_value(sb,1)
    order=['B','A'] if mode in ['serial_ba','parallel_ba'] else ['A','B']
    if mode=='a_alone':order=['A']
    if mode=='b_alone':order=['B']
    if mode=='empty_control':b=backend.empty_graph()
    serial=mode in ['serial_ab','serial_ba','non_overlap_control']
    stream={'A':sa,'B':sb}
    if mode=='same_stream_control':stream['B']=sa
    for i,role in enumerate(order):
        if serial and i:
            # The producer's event record has already been enqueued; waiting
            # on an event that has never been recorded is not a start latch.
            backend.wait_event(stream[role],order[i-1]+'_end')
        backend.record(role+'_start',stream[role])
        backend.launch(a if role=='A' else b,stream[role])
        backend.record(role+'_end',stream[role])
    # Both streams' work was enqueued before releasing either branch.
    backend.record('joint_start',ctrl)
    backend.write_value(ctrl,1)
    for role in order:backend.wait_event(ctrl,role+'_end')
    backend.record('joint_end',ctrl)
    backend.synchronize_event('joint_end')
    return dict(order=order,measurement_scope='GPU event envelopes; kernel intervals require separate P2',
                release_after_all_submissions=True,simultaneous_gpu_start_claim=False)
