"""Summarize real Decode node intervals separately from paired unprofiled model time."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean,median,stdev

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
MODES=('original','sham','fork')

def read(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)

def write(name,rows):
    if not rows:return
    OUT.mkdir(exist_ok=True)
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def p95(xs):
    xs=sorted(xs)
    i=(len(xs)-1)*.95
    return xs[int(i)]+(i-int(i))*(xs[min(int(i)+1,len(xs)-1)]-xs[int(i)])

def timeline(t,fields):
    steps=[]
    bymode=defaultdict(list)
    for r in t['replays']:
        g=r['layers']
        if len(g)!=36:raise ValueError('incomplete replay')
        durations=[]
        for layer,p,qs,qe,ks,ke,c,att,_,_ in g:
            durations.append(dict(q_us=(qe-qs)/1000,k_us=(ke-ks)/1000,overlap_us=max(0,min(qe,ke)-max(qs,ks))/1000,
                  start_delta_us=(ks-qs)/1000,pair_us=(max(qe,ke)-min(qs,ks))/1000,
                  pre_start_us=(min(qs,ks)-p)/1000,join_us=(c-max(qe,ke))/1000,
                  producer_to_cache_us=(c-p)/1000))
        bymode[r['mode']].extend(durations)
        steps.append(dict(**fields,mode=r['mode'],step=r['step'],graph_ms=r['graph_ns']/1e6,
                          sum_pair_us=sum(row['pair_us'] for row in durations),
                          sum_gap_and_hideable_us=sum(row['pair_us']-max(row['q_us'],row['k_us']) for row in durations),
                          sum_serial_hideable_us=sum(min(row['q_us'],row['k_us']) for row in durations),
                          sum_producer_to_cache_us=sum(row['producer_to_cache_us'] for row in durations)))
    totals=[]
    for mode,rows in bymode.items():
        s=[x for x in steps if x['mode']==mode]
        separate=sum(qs!=ks for r in t['replays'] if r['mode']==mode for *_,qs,ks in r['layers'])
        totals.append(dict(**fields,mode=mode,replays=len(s),layer_pairs=len(rows),overlap_hits=sum(x['overlap_us']>0 for x in rows),
                          distinct_stream_pairs=separate,
                          **{k+'_median':median(x[k] for x in rows) for k in rows[0]},
                          graph_median_ms=median(x['graph_ms'] for x in s),sum_pair_median_us=median(x['sum_pair_us'] for x in s),
                          sum_gap_and_hideable_median_us=median(x['sum_gap_and_hideable_us'] for x in s),
                          sum_serial_hideable_median_us=median(x['sum_serial_hideable_us'] for x in s),
                          sum_producer_to_cache_median_us=median(x['sum_producer_to_cache_us'] for x in s)))
    return steps,totals

def analyze():
    inv=[];trials=[];steps=[];timings=[];paired=[];est=[];telemetry=[];triplets=[];resources=[]
    for p in sorted((ROOT/'raw').iterdir()):
        guard=p/'supervision.json'
        if not guard.exists():continue
        guard=json.loads(guard.read_bytes())
        f=dict(run=p.name,precision=p.name.split('_')[-2],rep=p.name.split('_')[-3],profile=p.name.split('_')[-1].upper())
        telemetry.append(dict(**f,start=guard['start'],end=guard['end'],status=guard['status'],error=guard.get('error'),gpu_started=guard['gpu_workload_started']))
        if guard['status']!='COMPLETED':continue
        j=read(p/'inference.json.gz')
        hist=next((ROOT.parent/'gate4_targeted/raw').glob('*_r2_'+f['precision']+'_p2/inference.json.gz'))
        baseline=read(hist)
        def outputs(x):
            return sorted((r['prompt_hash'],r['output_hash'],r['prompt_length'],r['output_length']) for r in x['request_summary'])
        if j['invocation_status']!='COMPLETED' or j['config_file_sha256']!=baseline['config_file_sha256'] or len(j['branches']['layers'])!=36:
            raise ValueError('identity mismatch')
        if any(outputs(t)!=outputs(baseline['runs'][0]) or len(t['graph_replays'])!=256 for t in j['runs']):
            raise ValueError('historical request hashes or steps differ')
        b=j['branches']
        inv.append(dict(**f,full_requests_groups=len(j['runs']),layers=len(b['layers']),original_kernels=b['kernels'],
                        input_manifest=j['prompt_manifest_hash'],warm_replays=b['warm_replays'],outputs_match_history=True,
                        model_ms_mean=mean(t['full_model_wall_ms'] for t in j['runs'])))
        if f['profile']=='P2':
            trace=read(p/'intervals.json.gz')
            if trace['run']!=p.name or trace['uuid']!=j['gpu_uuid'].removeprefix('GPU-'):raise ValueError('wrong trace')
            for key,entry in trace['launch_resources'].items():
                layer,role=key.split('_')
                resources.append(dict(**f,layer=int(layer),role=role,**entry))
            s,tt=timeline(trace,f);steps+=s;timings+=tt
            for ix in range(0,len(s)-2,3):
                bb={x['mode']:x for x in s[ix:ix+3]}
                if set(bb)!=set(MODES):raise ValueError('invalid three-step balanced sequence')
                a,b=bb['fork'],bb['sham']
                triplets.append(dict(**f,first_step=s[ix]['step'],fork_step=a['step'],sham_step=b['step'],
                                graph_change_us=(a['graph_ms']-b['graph_ms'])*1000,
                                sum_pair_change_us=a['sum_pair_us']-b['sum_pair_us'],
                                sum_consumer_wait_change_us=a['sum_producer_to_cache_us']-b['sum_producer_to_cache_us']))
            continue
        if len(j['runs'])!=15:raise ValueError('incomplete P0')
        values=defaultdict(dict)
        for i,(mode,t) in enumerate(zip(b['p0_modes'],j['runs'],strict=True)):
            gaps=[]
            for req in t['requests'].values():
                ev=req['output_events']
                if len(ev)!=257 or any(x['count']!=1 for x in ev):raise ValueError('incomplete token events')
                gaps.extend((y['time_ns']-x['time_ns'])/1e6 for x,y in zip(ev,ev[1:]))
            row=dict(**f,block=i//3,trial=i,mode=mode,model_ms=t['full_model_wall_ms'],
                     joint_ms=(t['post_response_gpu_drain_end_ns']-t['full_cpu_start_ns'])/1e6,
                     ttft_ms=mean(x['ttft_ms'] for x in t['request_summary']),
                     tpot_ms=mean(x['tpot_ms'] for x in t['request_summary']),itl_p95_ms=p95(gaps))
            trials.append(row);values[i//3][mode]=row
        gg=defaultdict(list)
        for n,bb in values.items():
            if set(bb)!=set(MODES):raise ValueError('incomplete pair')
            for control in ('sham','original'):
                for metric in ('model_ms','joint_ms','ttft_ms','tpot_ms','itl_p95_ms'):
                    fc,sc=bb['fork'][metric],bb[control][metric]
                    g=dict(**f,block=n,control=control,metric=metric,ms_diff=fc-sc,pct_diff=100*(fc/sc-1))
                    gg[(control,metric)].append(g);paired.append(g)
        for (control,metric),records in gg.items():
            row=dict(**f,control=control,metric=metric,blocks=len(records))
            for key in ('ms_diff','pct_diff'):
                a=[v[key] for v in records]
                center=mean(a);half=2.776445*stdev(a)/len(a)**.5
                row.update({key:center,key+'_ci95_lo':center-half,key+'_ci95_hi':center+half})
            est.append(row)
    scales=[]
    for base in (r for r in timings if r['mode']=='sham'):
        key=(base['precision'],base['rep'])
        fork=next(r for r in timings if (r['precision'],r['rep'],r['mode'])==(*key,'fork'))
        control=[r['model_ms'] for r in trials if (r['precision'],r['rep'],r['mode'])==(*key,'sham')]
        if not control:continue
        denominator=median(control)
        optimistic=256*base['sum_gap_and_hideable_median_us']/1000
        extrapolated=256*(fork['sum_pair_median_us']-base['sum_pair_median_us'])/1000
        scales.append(dict(precision=key[0],rep=key[1],p0_sham_median_ms=denominator,
                          idealized_saved_ms=optimistic,idealized_saved_pct=100*optimistic/denominator,
                          p2_pair_cost_change_extrapolated_ms=extrapolated,
                          p2_pair_cost_change_pct=100*extrapolated/denominator))
    for n,rows in [('inventory',inv),('supervision',telemetry),('p2_steps',steps),('p2_summary',timings),('p2_triplets',triplets),('launch_resources',resources),('whole_request_scale',scales),
                   ('p0_trials',trials),('paired_blocks',paired),('p0_estimates',est)]:
        write(n,rows)
if __name__=='__main__':
    analyze()
