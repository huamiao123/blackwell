"""Keep P0 trial pairs, P2 kernel intervals, and independent processes distinct."""
from collections import defaultdict
import csv
import gzip
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean,median,stdev
from protocol import MODES,STEPS,schedules

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
def read(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)
def save(name,rows):
    if not rows:return
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n')
        w.writeheader();w.writerows(rows)
def quantile(data,q):
    v=sorted(data);x=(len(v)-1)*q;t=int(x)
    return v[t]+(v[min(t+1,len(v)-1)]-v[t])*(x-t)
def tokens(trial):
    r=trial['request_summary']
    if len({i['prompt_hash'] for i in r})!=8 or any((i['prompt_length'],i['output_length'])!=(2048,257) for i in r):
        raise ValueError('incorrect request shape')
    return sorted((x['prompt_hash'],x['output_hash']) for x in r)
@lru_cache(maxsize=2)
def baseline(prec):
    suffix='073642' if prec=='nvfp4' else '073813'
    f,=(ROOT.parent/'gate4_inference/raw/dram_20261119').glob('*'+suffix+'*/*.json.gz')
    j=read(f)
    return j['config_file_sha256'],j['prompt_manifest_hash'],tokens(j['runs'][0])
def check(p):
    j=read(p/'inference.json.gz')
    prec=j['config']['precision']
    cfg,manifest,outputs=baseline(prec)
    if (j['invocation_status'],j['config_file_sha256'],j['prompt_manifest_hash'])!=('COMPLETED',cfg,manifest):
        raise ValueError('wrong config or result')
    meta=j['target_nodes']
    seed=84239 if '_r2_' in p.name else 71011
    plan=schedules(j['profile'],seed)
    if len(plan)!=len(j['runs']) or len(plan)!=len(meta['trials']):raise ValueError('incomplete trial count')
    if meta['work']['flops']!=268435456 or meta['work']['nonempty_ctas']!=64:raise ValueError('unequal workload')
    if j['profile']=='P0' and meta.get('warm_variant_launches',0)<16:raise ValueError('P0 variants not prewarmed')
    if meta['probe_sha256']!=json.loads((ROOT.parent/'gate4_spatial/analysis/offline_compile/compile_manifest.json').read_bytes())['cubin_sha256']:
        raise ValueError('wrong probe binary')
    for trial,checks,modes in zip(j['runs'],meta['trials'],plan):
        if (len(trial['graph_replays']),trial['total_output_tokens'],tokens(trial))!=(256,2056,outputs):
            raise ValueError('E2 token baseline mismatch')
        for i,c,mode in zip(STEPS,checks,modes,strict=True):
            if (c['iteration'],c['mode'])!=(i,mode):raise ValueError('schedule mismatch')
            if mode not in ('original','sham') and not (c['correctness_passed'] and c['elements_checked']==65536):
                raise ValueError('output validation failed')
    return j
def analyze():
    inventory=[];p2=[];p0=[];stats=[];pairs=[];p2summary=[]
    for p in sorted((ROOT/'raw').iterdir()):
        if not (p/'inference.json.gz').exists():continue
        j=check(p)
        prec=j['config']['precision'];meta=j['target_nodes'];rep=p.name.split('_')[-3]
        fields=dict(run_id=p.name,precision=prec,rep=rep)
        inv=dict(**fields,profile=j['profile'],trials=len(j['runs']),outputs_match_history=True,
                 independent_process=True,graph_replays_per_trial=256,verified_probe_flops=meta['work']['flops'],
                 probe_work_hash=meta['work']['work_hash'],warm_variant_launches=meta.get('warm_variant_launches',0),
                 config_sha256=j['config_file_sha256'],prompt_hash=j['prompt_manifest_hash'])
        inventory.append(inv)
        if j['profile']=='P2':
            trace=json.loads((p/'validated_trace.json').read_bytes())
            if trace['gpu_uuid']!=j['gpu_uuid'].removeprefix('GPU-'):raise ValueError('trace identity mismatch')
            rows=trace['rows'];p2.extend(dict(**fields,**r) for r in rows)
            for op in ('o','down'):
                g=[r for r in rows if r['mode']==op+'_parallel']
                sham=[r for r in rows if r['mode']=='sham']
                ab=[r['b_ns'] for r in rows if r['mode']==op+'_ab']
                ba=[r['b_ns'] for r in rows if r['mode']==op+'_ba']
                def pair_span(r):
                    return max(r['a_end_ns'],r['b_end_ns'])-min(r['a_start_ns'],r['b_start_ns'])
                parallel_span=sum(pair_span(r) for r in g)
                ab_span=sum(pair_span(r) for r in rows if r['mode']==op+'_ab')
                ba_span=sum(pair_span(r) for r in rows if r['mode']==op+'_ba')
                b=sum(r['b_ns'] for r in g)
                a=median(r[op+'_ns'] for r in g);a0=median(r[op+'_ns'] for r in sham)
                p2summary.append(dict(**fields,operator=op,n=len(g),overlap_hits=sum(r['overlap_ns']>0 for r in g),
                    fully_inside=sum(r['fully_inside'] for r in g),fraction_b=sum(r['overlap_ns'] for r in g)/b,
                    a_us=a/1000,sham_a_us=a0/1000,b_us=median(r['b_ns'] for r in g)/1000,
                    preflight_event_us=meta['baseline']['median_us'],b_rate_vs_ab=sum(ab)/b,b_rate_vs_ba=sum(ba)/b,
                    local_ab_over_parallel=ab_span/parallel_span,local_ba_over_parallel=ba_span/parallel_span,
                    start_delta_us=median(r['start_delta_ns'] for r in g)/1000,
                    end_margin_us=median(r['finish_margin_ns'] for r in g)/1000))
            continue
        blocks=defaultdict(dict)
        for idx,(trial,cs) in enumerate(zip(j['runs'],meta['trials'])):
            mode=cs[0]['mode'];n=idx//len(MODES)
            gaps=[]
            for request in trial['requests'].values():
                ev=request['output_events']
                if len(ev)!=257 or any(e['count']!=1 for e in ev):raise ValueError('invalid ITL series')
                gaps.extend((ev[s]['time_ns']-ev[s-1]['time_ns'])/1e6 for s in STEPS)
            row=dict(**fields,block=n,trial=idx,mode=mode,model_ms=trial['full_model_wall_ms'],
                     joint_ms=(trial['post_response_gpu_drain_end_ns']-trial['full_cpu_start_ns'])/1e6,
                     itl_mean_ms=mean(gaps),itl_p95_ms=quantile(gaps,.95),request_max_ms=max(t['model_request_ms'] for t in trial['request_summary']),
                     ttft_mean_ms=mean(t['ttft_ms'] for t in trial['request_summary']),
                     tpot_mean_ms=mean(t['tpot_ms'] for t in trial['request_summary']),
                     probe_count=sum('output_sha256' in c for c in cs))
            blocks[n][mode]=row;p0.append(row)
        bymetric=defaultdict(list)
        for n,conditions in blocks.items():
            if set(conditions)!=set(MODES):raise ValueError('incomplete paired block')
            for op in ('o','down'):
                a=conditions[op+'_parallel'];sham=conditions['sham'];orig=conditions['original']
                metrics=dict(model_loss_vs_original_pct=100*(a['model_ms']/orig['model_ms']-1),
                             model_loss_vs_sham_pct=100*(a['model_ms']/sham['model_ms']-1),
                             target_itl_p95_vs_sham_pct=100*(a['itl_p95_ms']/sham['itl_p95_ms']-1),
                             serial_ab_over_parallel=conditions[op+'_ab']['joint_ms']/a['joint_ms'],
                             serial_ba_over_parallel=conditions[op+'_ba']['joint_ms']/a['joint_ms'])
                for k,v in metrics.items():
                    bymetric[op,k].append(v)
                    pairs.append(dict(**fields,block=n,op=op,metric=k,value=v))
            for k,v in dict(down_minus_o_parallel_joint_ms=conditions['down_parallel']['joint_ms']-conditions['o_parallel']['joint_ms'],
                            sham_minus_original_ms=conditions['sham']['model_ms']-conditions['original']['model_ms']).items():
                bymetric['paired',k].append(v);pairs.append(dict(**fields,block=n,op='paired',metric=k,value=v))
        for (op,k),vs in bymetric.items():
            if len(vs)!=5:raise ValueError('five blocks required')
            a=mean(vs);h=2.776445*stdev(vs)/len(vs)**.5
            stats.append(dict(**fields,op=op,metric=k,blocks=5,mean=a,minimum=min(vs),maximum=max(vs),ci95_low=a-h,ci95_high=a+h))
    predicted=[]
    with (ROOT.parent/'gate4_microarch/analysis/full_decode_duration_bounds.csv').open() as f:
        historical={(r['precision'],r['operator'],r['layer']):r for r in csv.DictReader(f)}
    for r in p2summary:
        h=historical[r['precision'],r['operator']+'_proj','18']
        forecast=float(h['p50_us'])-float(h['isolated_b_alone_us'])
        predicted.append(dict(run_id=r['run_id'],precision=r['precision'],operator=r['operator'],
                             archived_layer18_p50_us=h['p50_us'],archived_probe_alone_us=h['isolated_b_alone_us'],
                             forecast_margin_us=forecast,observed_median_margin_us=r['end_margin_us'],
                             error_us=r['end_margin_us']-forecast,overlap_hits=r['overlap_hits'],fully_inside=r['fully_inside'],targeted=r['n']))
    scales=[]
    for r in p2summary:
        prec,rep,op=r['precision'],r['rep'],r['operator']
        original=[v['model_ms'] for v in p0 if (v['precision'],v['rep'],v['mode'])==(prec,rep,'original')]
        if not original:continue
        serial=[v['b_ns']/1000 for v in p2 if v['precision']==prec and v['rep']==rep and v['mode'] in (op+'_ab',op+'_ba')]
        full=median(original)
        demand_ms=len(STEPS)*median(serial)/1000
        tail_ms=len(STEPS)*max(0,-r['end_margin_us'])/1000
        scales.append(dict(precision=prec,process_rep=rep,operator=op,injections=len(STEPS),
                           serial_b_kernel_median_us=median(serial),p0_original_median_ms=full,
                           extrapolated_b_time_ms=demand_ms,extrapolated_b_pct_of_request=100*demand_ms/full,
                           extrapolated_exposed_tail_ms=tail_ms,extrapolated_tail_pct_of_request=100*tail_ms/full))
    for name,rows in [('run_inventory',inventory),('p2_nodes',p2),('p2_summary',p2summary),('p0_trials',p0),('paired_blocks',pairs),('p0_summary',stats),('forecast_check',predicted),('request_scale',scales)]:
        save(name+'.csv',rows)
if __name__=='__main__':analyze()
