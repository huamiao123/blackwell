"""CPU-only data preparation with real tokenizer + Chat Template, no padding IDs."""
import argparse
import os
from pathlib import Path
import re
import importlib.metadata
from common import ROOT,digest,file_hash,load,save,csv_write


def load_tokenizer(path):
    os.environ.update(USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)


def render(tok,messages):
    return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)


def encode(tok,text):
    return tok.encode(text,add_special_tokens=False)


def calibrate(tok,source,target,instruction,dialogue=False):
    """Crop contiguous source tokens; no repeated padding text. Check final template length."""
    source_ids=encode(tok,source[:max(65536,target*16)])
    def make(n):
        excerpt=tok.decode(source_ids[:n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        messages=[dict(role='user',content=instruction+'\n\n'+excerpt)]
        if dialogue:
            messages.extend([dict(role='assistant',content='I have read the excerpt. I will distinguish textual evidence from interpretation.'),
                             dict(role='user',content='Now explain the central conflict, cite the relevant details, and identify one uncertainty.')])
        text=render(tok,messages)
        return messages,text,encode(tok,text)
    lo,hi=0,len(source_ids)
    while lo<=hi:
        mid=(lo+hi)//2
        m,t,ids=make(mid)
        if len(ids)==target:return m,t,ids,mid
        if len(ids)<target:lo=mid+1
        else:hi=mid-1
    for n in range(max(0,lo-96),min(len(source_ids),lo+96)+1):
        m,t,ids=make(n)
        if len(ids)==target:return m,t,ids,n
    raise ValueError(('exact natural excerpt length unavailable',target,len(source_ids),instruction))


def prepare(tokenizer_dir):
    if any((ROOT/'raw').glob('*.json')):raise RuntimeError('Dataset frozen after real runs; create a new version')
    tok=load_tokenizer(tokenizer_dir)
    valid=set(tok.get_vocab().values());special=set(tok.all_special_ids)
    meta=dict(tokenizer_version=importlib.metadata.version('transformers'),
              tokenizer_file_hash=file_hash(Path(tokenizer_dir)/'tokenizer.json'),
              tokenizer_config_hash=file_hash(Path(tokenizer_dir)/'tokenizer_config.json'),
              chat_template_hash=digest(tok.chat_template),defined_id_count=len(valid),
              max_defined_id=max(valid),special_ids=sorted(special),canonical_tokenizer='Qwen3-8B-BF16')
    save(ROOT/'prompts/manifests/tokenizer_contract.json',meta,overwrite=True)
    sources=load(ROOT/'prompts/sources/source_manifest.json')
    by_id={s['id']:s for s in sources}
    specs=[('pride','general_qa',0),('sherlock','general_qa',18000),
           ('heapq','code',0),('statistics','code',0),
           ('pride','long_document',240000),('moby','long_document',40000),
           ('pride','multi_turn',90000),('sherlock','multi_turn',130000)]
    records=[]
    def add(category,r,length,source,source_info,instruction,dialogue=False):
        messages,text,ids,n=calibrate(tok,source,length,instruction,dialogue)
        for msg in messages:
            if set(encode(tok,msg['content']))&special:raise ValueError('special token inside message body')
        if not set(ids)<=valid:raise ValueError('undefined tokenizer ID')
        row=dict(prompt_id=f'{category}_r{r}_l{length}',prompt_category=category,raw_text=text,messages=messages,
                 token_ids=ids,token_ids_sha256=digest(ids),actual_token_length=len(ids),target_length=length,
                 special_token_positions=[dict(position=i,token_id=x) for i,x in enumerate(ids) if x in special],
                 bos_policy='no automatic BOS; add_special_tokens=False',eos_policy='template boundaries only; generation policy in config',
                 thinking_mode=False,add_generation_prompt=True,prefix_hash=digest(ids[:32]),
                 data_source=source_info,source_tokens_used=n,source_crop='contiguous token prefix; may end mid-sentence',
                 language='en',**meta)
        dest=ROOT/'prompts'/('controlled' if category=='synthetic_control' else 'realistic')/(row['prompt_id']+'.json')
        save(dest,row,overwrite=True)
        records.append(dict(prompt_id=row['prompt_id'],path=dest.relative_to(ROOT).as_posix(),file_sha256=file_hash(dest),
                            category=category,length=len(ids),token_ids_sha256=row['token_ids_sha256']))
    for r in range(8):
        # This is intentionally synthetic; it is never relabelled as natural text.
        text='\n'.join(f'Experiment {r}: record {i}, identifier {(r+3)*100003+i*97}, observed value {(i*37+r*17)%997}.' for i in range(1500))
        for length in [128,512,2048,4096,8192]:
            add('synthetic_control',r,length,text,dict(kind='self_authored_deterministic',generator='numbered experimental records v1',row=r),
                f'Control sample {r}: describe the numbered records without treating them as natural observations.')
    for r,(sid,category,offset) in enumerate(specs):
        info=by_id[sid];path=ROOT/info['local_path']
        if file_hash(path)!=info['sha256']:raise ValueError('source changed')
        text=path.read_text(encoding=info.get('encoding','utf-8-sig'))
        # Use the archived source bytes. Keep its legal notice in the source file.
        start=text.find('*** START OF')
        if start>=0:
            start=text.find('\n',text.find('***',start+4))+1
            text=text[start:]
        text=text[offset:]
        instruction=(f'Case {r}, {info["title"]}: explain the excerpt using only evidence present in it. '
                     'For incomplete code or prose, state the limitation rather than inventing the missing context.')
        for length in [128,256,512,1024,2048,4096]:
            add(category,r,length,text,dict(**info,character_offset_after_header=offset),instruction,category=='multi_turn')
    manifest=dict(schema='v4.1',records=records,tokenizer=meta,
                  dataset_limit='Small reproducible English reading/code/dialogue benchmark, not production traffic or broad task quality evaluation')
    manifest['manifest_sha256']=digest(records)
    save(ROOT/'prompts/manifests/dataset.json',manifest,overwrite=True)
    csv_write(ROOT/'analysis/prompt_manifest.csv',records)
    return manifest


def configs(manifest):
    def matching(category,length):
        return [r for r in manifest['records'] if (r['category']=='synthetic_control')==category and r['length']==length]
    groups={'E1':matching(True,2048)[:1],'E2':matching(True,2048),
            'E3':matching(False,512)[:1],'E4':matching(False,512),
            'E5':matching(False,2048)[:1],'E6':matching(False,2048)}
    natural=[r for r in manifest['records'] if r['category']!='synthetic_control']
    # E7 is a specified research distribution, never a production-flow claim.
    lengths=[128,256,512,512,1024,2048,2048,4096]
    # 256/1024 records are prepared separately by the same calibration function.
    available={r['prompt_id']:r for r in natural}
    e7=[]
    for row,length in enumerate(lengths):
        matches=[r for r in natural if f'_r{row}_' in r['prompt_id'] and r['length']==length]
        if not matches:continue
        e7.append(matches[0])
    if len(e7)==8:groups['E7']=e7
    for name,rs in groups.items():
        for prec in ['bf16','nvfp4']:
            cfg=dict(schema='v4.1',experiment=name,precision=prec,batch=len(rs),output_mode='fixed_decode',decode_steps=256,
                     max_output_tokens=257,trials=5,warmup_trials=2,warmup_output_tokens=33,
                     prompt_paths=[r['path'] for r in rs],dataset_manifest_sha256=manifest['manifest_sha256'],
                     prompt_work_hash=digest([r['token_ids_sha256'] for r in rs]),seed=0,temperature=0.,ignore_eos=True,
                     generation_config='vllm',stop_token_ids=[],scope='offline_generate_with_preencoded_prompts',
                     engine=dict(max_model_len=4608,gpu_memory_utilization=.85,linear_backend='cutlass',
                                 enforce_eager=False,kv_cache_dtype='fp8_e4m3',attention_backend='FLASHINFER',
                                 disable_log_stats=True,enable_prefix_caching=False,enable_chunked_prefill=False,
                                 max_num_batched_tokens=16384,max_num_seqs=8,cudagraph_capture_sizes=[1,8],seed=0))
            cfg['config_hash']=digest(cfg)
            save(ROOT/'configs'/(name+'_'+prec+'.json'),cfg,overwrite=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-dir',required=True);a=ap.parse_args()
    m=prepare(a.tokenizer_dir);configs(m);print('PREPARED',len(m['records']),'CPU-tokenized prompts')
