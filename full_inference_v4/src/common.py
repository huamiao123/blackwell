"""CPU-only contracts shared by data preparation, workers and analysis."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MISSING = {'UNAVAILABLE', 'UNVERIFIED', 'NOT_APPLICABLE'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if overwrite else 'x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def percentile(values, q):
    if not values:
        return 'UNAVAILABLE'
    s = sorted(values)
    p = (len(s)-1)*q
    a = int(p)
    return s[a] + (s[min(a+1,len(s)-1)]-s[a])*(p-a)


def union(intervals):
    result = []
    for s, e in sorted(intervals):
        if e <= s: continue
        if result and s <= result[-1][1]: result[-1][1] = max(e,result[-1][1])
        else: result.append([s,e])
    return result


def intersection(a,b):
    a,b=union(a),union(b)
    i=j=0;out=[]
    while i<len(a) and j<len(b):
        s,e=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if s<e:out.append([s,e])
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return out


def duration(xs):
    return sum(e-s for s,e in union(xs))


def classify_schedule(before, scheduled):
    """Use PRE-schedule counters, never optimistic post-schedule counters."""
    requests=[]
    for rid,n in scheduled.items():
        if n<0: raise ValueError('negative scheduled tokens')
        computed,prompt=before[rid]['computed'],before[rid]['prompt_length']
        prefill=min(n,max(0,prompt-computed))
        requests.append(dict(request_id=rid,computed_before=computed,prompt_length=prompt,
                             scheduled_tokens=n,prefill_tokens=prefill,decode_tokens=n-prefill))
    p=sum(r['prefill_tokens'] for r in requests)
    d=sum(r['decode_tokens'] for r in requests)
    phase='MIXED' if p and d else 'PREFILL' if p else 'DECODE' if d else 'EMPTY'
    return dict(stage=phase,scheduled_prefill_tokens=p,scheduled_decode_tokens=d,
                active_request_count=len([r for r in requests if r['scheduled_tokens']]),requests=requests)


def validate_config(config, prompts):
    if config['precision'] not in ['bf16','nvfp4']:raise ValueError('precision')
    if len(prompts)!=config['batch']:raise ValueError('batch')
    if len({p['token_ids_sha256'] for p in prompts})!=len(prompts):raise ValueError('duplicate prompts')
    if config['output_mode']=='fixed_decode':
        needed=config['decode_steps']+1
        if config['max_output_tokens']!=needed:raise ValueError('prefill output plus Decode outputs')
    else:needed=config['max_output_tokens']
    if any(p['actual_token_length']+needed>config['engine']['max_model_len'] for p in prompts):
        raise ValueError('insufficient context budget')
    if not config['engine']['enable_chunked_prefill'] and sum(p['actual_token_length'] for p in prompts)>config['engine']['max_num_batched_tokens']:
        raise ValueError('unchunked prefill token budget')
    if config['engine']['enable_prefix_caching']:raise ValueError('core baseline must disable prefix caching')
    if config['engine']['enforce_eager']:raise ValueError('core Graph baseline')
    if config['engine']['kv_cache_dtype']!='fp8_e4m3':raise ValueError('KV dtype')
    return True


def paired_metrics(a, b, serial_ab, serial_ba, parallel):
    # All inputs represent the same fixed work, not independently calibrated durations.
    for row in [b,serial_ab,serial_ba,parallel]:
        if row['b_work_hash']!=b['b_work_hash']:raise ValueError('unequal B work')
    for row in [serial_ab,serial_ba,parallel]:
        if row['a_work_hash']!=a['a_work_hash']:raise ValueError('unequal A work')
    return dict(ia=parallel['a_ms']/a['a_ms']-1,
                rb=b['b_span_ms']/parallel['b_span_ms'],
                s_ab=serial_ab['joint_ms']/parallel['joint_ms'],
                s_ba=serial_ba['joint_ms']/parallel['joint_ms'],
                s_fastest_serial=min(serial_ab['joint_ms'],serial_ba['joint_ms'])/parallel['joint_ms'],
                b_complete_before_a_end=parallel['b_end_ns']<=parallel['a_end_ns'])
