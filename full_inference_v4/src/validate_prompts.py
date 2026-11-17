"""CPU semantic validation: defined IDs, template-only specials, exact encoding."""
import argparse
import copy
import sys
from pathlib import Path
from common import ROOT,load,digest,validate_config,save,file_hash
from prepare_prompts import load_tokenizer,render,encode


def validate_one(p,tok):
    valid=set(tok.get_vocab().values());special=set(tok.all_special_ids)
    ids=p['token_ids']
    if not set(ids)<=valid:raise ValueError('undefined token ID')
    if len(ids)!=p['actual_token_length'] or len(ids)!=p['target_length']:raise ValueError('length')
    if digest(ids)!=p['token_ids_sha256']:raise ValueError('token hash')
    for m in p['messages']:
        if set(encode(tok,m['content']))&special:raise ValueError('special in message body')
    if p['thinking_mode'] is not False:raise ValueError('thinking policy changed')
    if render(tok,p['messages'])!=p['raw_text']:raise ValueError('Chat Template mismatch')
    if encode(tok,p['raw_text'])!=ids:raise ValueError('encoding mismatch')
    if p['special_token_positions']!=[dict(position=i,token_id=x) for i,x in enumerate(ids) if x in special]:
        raise ValueError('special positions')
    return True


def lcp(rows):
    n=0
    for values in zip(*(p['token_ids'] for p in rows)):
        if len(set(values))!=1:break
        n+=1
    return n


def validate_all(tokenizer_dir):
    tok=load_tokenizer(tokenizer_dir);m=load(ROOT/'prompts/manifests/dataset.json')
    if digest(m['records'])!=m['manifest_sha256']:raise ValueError('manifest hash')
    if file_hash(Path(tokenizer_dir)/'tokenizer.json')!=m['tokenizer']['tokenizer_file_hash']:raise ValueError('tokenizer asset hash')
    if file_hash(Path(tokenizer_dir)/'tokenizer_config.json')!=m['tokenizer']['tokenizer_config_hash']:raise ValueError('tokenizer config hash')
    for r in m['records']:
        if file_hash(ROOT/r['path'])!=r['file_sha256']:raise ValueError('prompt file hash')
        validate_one(load(ROOT/r['path']),tok)
    configs=[]
    for f in sorted((ROOT/'configs').glob('E*.json')):
        c=load(f);rs=[load(ROOT/p) for p in c['prompt_paths']]
        expected=c.pop('config_hash');assert digest(c)==expected;c['config_hash']=expected
        validate_config(c,rs)
        shared=lcp(rs) if len(rs)>1 else 0
        if len(rs)>1 and shared>32:raise ValueError('unexpected shared prefix')
        configs.append(dict(config=f.name,batch=len(rs),lengths=[p['actual_token_length'] for p in rs],
                            common_prefix_tokens=shared,status='CPU_VALIDATED'))
    result=dict(scope='CPU real-tokenizer validation only',gpu_started=False,prompts=len(m['records']),
                categories=sorted({r['category'] for r in m['records']}),configs=configs,
                dataset_hash=m['manifest_sha256'],status='VERIFIED_CPU_ONLY')
    # Independent negative cases use the actual tokenizer, not a fake vocabulary.
    original=load(ROOT/m['records'][0]['path']);tests=[]
    for kind in ['undefined_id','special_in_body','wrong_length','wrong_hash','wrong_template']:
        bad=copy.deepcopy(original)
        if kind=='undefined_id':bad['token_ids'][0]=max(tok.get_vocab().values())+1
        elif kind=='special_in_body':bad['messages'][0]['content']+='<|im_start|>'
        elif kind=='wrong_length':bad['actual_token_length']+=1
        elif kind=='wrong_hash':bad['token_ids_sha256']='invalid'
        else:bad['raw_text']+=' unexpected change'
        try:validate_one(bad,tok)
        except ValueError:tests.append(kind)
        else:raise AssertionError('negative test accepted: '+kind)
    result['real_tokenizer_negative_tests_passed']=tests
    result['torch_imported_during_cpu_tokenization']='torch' in sys.modules
    # Indirect imports can still load torch; this checks its lazy-init flag only,
    # never cuda.is_available(), device discovery or a CUDA allocation.
    result['cuda_initialized_during_cpu_tokenization']=(sys.modules['torch'].cuda.is_initialized() if 'torch' in sys.modules else False)
    if result['cuda_initialized_during_cpu_tokenization']:raise RuntimeError('CPU tokenizer unexpectedly initialized CUDA')
    old=ROOT.parent/'resource_overlap_v1/raw/gate1_nvfp4_b8_s2.json'
    if old.exists():
        ids=[x for row in load(old)['prompt_ids'] for x in row]
        valid=set(tok.get_vocab().values());special=set(tok.all_special_ids)
        result['old_b8_input_audit']=dict(undefined_positions=sum(x not in valid for x in ids),
                                          body_special_positions=sum(x in special for x in ids),old_file_sha256=file_hash(old))
    save(ROOT/'analysis/prompt_validation.json',result,overwrite=True)
    print('VALIDATED',len(m['records']),'prompts and',len(configs),'configs; GPU execution UNVERIFIED')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',required=True);a=p.parse_args();validate_all(a.tokenizer_dir)
