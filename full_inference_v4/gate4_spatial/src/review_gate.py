"""CPU entry guard and finite authorization contract. Current release is CPU-only."""
import datetime as dt
import os
from g4_common import load,sha
from contracts import ContractError

def review_status(config):
    if config.get('review_status')!='GPU_REVIEWED' or config.get('gpu_authorized') is not True:
        raise ContractError('GPU_DISABLED_CPU_PREPARATION: explicit review/window required')
    if config.get('auto_next_gate') or config.get('idle_polling'):raise ContractError('automatic advancement/polling forbidden')

def validate_window(window,task,now,hostname):
    if not window.get('enabled') or window.get('hostname')!=hostname:raise ContractError('window disabled/wrong host')
    if task not in window.get('allowed_tasks',[]):raise ContractError('task outside authorization')
    start,end=(dt.datetime.fromisoformat(window[k]) for k in ['not_before','expires_at'])
    if start.tzinfo is None or end.tzinfo is None or not start.timestamp()<=now<end.timestamp():
        raise ContractError('outside bounded GPU window')
    if end.timestamp()-start.timestamp()>1200 or not window.get('authorization_reference'):
        raise ContractError('unbounded/missing authorization')
    return end.timestamp()

def single_node_prerequisites(status):
    needed=['live_node_identity','safe_input_buffer_contract','output_reference','cache_protocol',
            'probe_gpu_correctness','same_context_streams','non_overlap_control_verified']
    missing=[k for k in needed if status.get(k)!='VERIFIED']
    if missing:raise ContractError('single-node experiment not ready: '+','.join(missing))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();c=load(a.config)
    if a.execute:
        review_status(c)
        # No generic CLI may load historical pointers or implicitly launch a
        # vLLM workload. A reviewed live adapter must supply all prerequisites.
        single_node_prerequisites(c.get('prerequisites',{}))
        raise SystemExit('LIVE_ADAPTER_REQUIRED: this CPU preparation CLI never launches CUDA')
    print('CPU review:',c.get('task'),'GPU disabled; no model/driver imports')

if __name__=='__main__':main()
