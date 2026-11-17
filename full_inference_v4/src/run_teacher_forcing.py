"""Teacher-forcing control PLAN only. No speculative GPU/sampler patch is enabled."""
import argparse
from common import ROOT,load,save,digest
from run_full_inference import prepare


def make_plan(prompts,continuations,defined_ids):
    plans=[]
    for p in prompts:
        tokens=continuations[p['token_ids_sha256']]
        if len(tokens)!=257 or not set(tokens)<=set(defined_ids):raise ValueError('reference continuation length or IDs')
        plans.append(dict(prompt_hash=p['token_ids_sha256'],forced_output_ids=tokens,continuation_hash=digest(tokens),
                          first_output_policy='prefill emits reference token 0',
                          decode_inputs=tokens[:-1],decode_input_hash=digest(tokens[:-1]),decode_steps=256))
    return dict(status='DESIGN_ONLY_GPU_EXECUTION_DISABLED',plans=plans,
                invariants=['one real prefill followed by 256 cached one-token Decode steps',
                            'force the next input token after the normal model forward, never replace Decode with one M=256 prefill',
                            'same forcing overhead and scheduler shape in BF16 and NVFP4',
                            'verify sampler-hook Graph compatibility and forced-ID consumption before any claim'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--continuations',required=True)
    p.add_argument('--defined-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:raise SystemExit('Teacher forcing remains DESIGN_ONLY pending reviewed vLLM sampler integration; no GPU launch.')
    _,prompts=prepare(a.config)
    save(a.output,make_plan(prompts,load(a.continuations),load(a.defined_ids)))
