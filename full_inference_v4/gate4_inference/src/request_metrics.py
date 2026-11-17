"""Per-request engine intervals, separate from online client E2E."""
from useful_work import ContractError

def request_metrics(request):
    ts=request['output_times_ns']
    if not ts or any(y<=x for x,y in zip(ts,ts[1:])):raise ContractError('nonmonotonic/empty output events')
    if len(ts)!=len(request['output_token_ids']):raise ContractError('bundled events are not per-token ITL')
    arrival=request['arrival_ns'];end=request['completion_ns']
    if arrival>ts[0] or end<ts[-1]:raise ContractError('invalid request boundary')
    itl=[y-x for x,y in zip(ts,ts[1:])]
    return dict(request_id=request['request_id'],ttft_ns=ts[0]-arrival,
                tpot_ns=(ts[-1]-ts[0])/(len(ts)-1) if len(ts)>1 else 'NOT_APPLICABLE',
                itl_ns=itl,e2e_engine_ns=end-arrival,output_tokens=len(ts),
                scope='preencoded local engine; excludes client network and tokenization')

def compare_same_inference(original,aware):
    keys=['work_hash','request_ids','required_output_lengths','auxiliary_required_units','consumer_ids']
    if any(original.get(k)!=aware.get(k) for k in keys):raise ContractError('different inference work')
    if not aware.get('auxiliary_consumed') or not aware.get('semantic_validation'):
        raise ContractError('no validated real consumer')
    if original['profile']!='P0' or aware['profile']!='P0':raise ContractError('not formal P0')
    if min(original['joint_ns'],aware['joint_ns'])<=0:raise ContractError('invalid inference interval')
    return dict(inference_speedup=original['joint_ns']/aware['joint_ns'],scope='same completed request + useful-work set')
