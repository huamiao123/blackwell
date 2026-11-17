"""CPU-only derivation from measured Gate-2 artifacts; original data unchanged."""
import csv,gzip,json,math,statistics,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import load,save,csv_write,duration,union


def read_activity(stem):
    with gzip.open(ROOT/'analysis'/(stem+'_activities.json.gz'),'rt',encoding='utf-8') as f:return json.load(f)


def main():
    sessions=[];request_rows=[];iteration_rows=[];raws=[]
    for path in sorted((ROOT/'raw').glob('gate2_*.json')):
        d=load(path)
        if d.get('schema')!='v4.1' or d.get('invocation_status')!='COMPLETED':continue
        raws.append(d);c=d['config']
        for run in d['runs']:
            assert len(run['requests'])==c['batch']
            assert all(len(q['output_token_ids'])==257 for q in run['requests'].values())
            counts={rid:[0,0] for rid in run['requests']}
            for it in run['scheduler_iterations']:
                assert not it['preempted']
                for r in it['requests']:
                    counts[r['request_id']][0]+=r['prefill_tokens'];counts[r['request_id']][1]+=r['decode_tokens']
                iteration_rows.append(dict(source=path.name,profile=d['profile'],experiment=c['experiment'],precision=c['precision'],trial=run['trial'],
                    iteration=it['scheduler_iteration'],stage=it['stage'],prefill_tokens=it['scheduled_prefill_tokens'],decode_tokens=it['scheduled_decode_tokens'],
                    batch=it['active_request_count'],replays=len(it['graph_replay_ids'])))
            assert all(pf==2048 and dec==256 for pf,dec in counts.values())
            assert len(run['graph_replays'])==256
            assert all(r['itl_scope']=='per-token output' for r in run['request_summary'])
            for r in run['request_summary']:
                request_rows.append(dict(source=path.name,profile=d['profile'],experiment=c['experiment'],precision=c['precision'],trial=run['trial'],**r))
        ttft=[statistics.mean(r['ttft_ms'] for r in run['request_summary']) for run in d['runs']]
        tpot=[statistics.mean(r['tpot_ms'] for r in run['request_summary']) for run in d['runs']]
        full=[run['full_model_wall_ms'] for run in d['runs']]
        ci=2.776445105*statistics.stdev(full)/math.sqrt(len(full)) if len(full)==5 else None
        sessions.append(dict(source=path.name,profile=d['profile'],experiment=c['experiment'],precision=c['precision'],n_runs=len(full),
            full_ms=statistics.mean(full),full_sd_ms=statistics.stdev(full) if len(full)>1 else None,
            within_process_ci95_low=statistics.mean(full)-ci if ci else None,within_process_ci95_high=statistics.mean(full)+ci if ci else None,
            ttft_ms=statistics.mean(ttft),tpot_ms=statistics.mean(tpot),
            itl_p95_request_median_ms=statistics.median(r['itl_p95_ms'] for run in d['runs'] for r in run['request_summary']),
            total_output_tokens_per_run=c['batch']*257,output_tokens_s=statistics.mean(c['batch']*257*1000/t for t in full),
            code_hash=hashlib.sha256(json.dumps(d['code_sha256'],sort_keys=True).encode()).hexdigest(),
            config_file_sha256=d['config_file_sha256'],dataset_hash=d['prompt_manifest_hash']))
    p0=defaultdict(list)
    for s in sessions:
        if s['profile']=='P0':p0[(s['experiment'],s['precision'])].append(s)
    aggregate=[];overheads=[]
    for (exp,prec),rr in sorted(p0.items()):
        aggregate.append(dict(experiment=exp,precision=prec,independent_processes=len(rr),valid_runs=sum(r['n_runs'] for r in rr),
            full_ms=statistics.mean(r['full_ms'] for r in rr),process_min_mean_ms=min(r['full_ms'] for r in rr),process_max_mean_ms=max(r['full_ms'] for r in rr),
            ttft_ms=statistics.mean(r['ttft_ms'] for r in rr),tpot_ms=statistics.mean(r['tpot_ms'] for r in rr),
            itl_p95_request_median_ms=statistics.median(r['itl_p95_request_median_ms'] for r in rr),
            output_tokens_s=statistics.mean(r['output_tokens_s'] for r in rr)))
    for s in sessions:
        if s['profile']=='P0':continue
        baseline=p0[(s['experiment'],s['precision'])]
        assert all(s[k]==r[k] for r in baseline for k in ['code_hash','config_file_sha256','dataset_hash'])
        full=statistics.mean(r['full_ms'] for r in baseline);ttft=statistics.mean(r['ttft_ms'] for r in baseline);tpot=statistics.mean(r['tpot_ms'] for r in baseline)
        overheads.append(dict(experiment=s['experiment'],precision=s['precision'],profile=s['profile'],source=s['source'],
                             full_overhead_percent=100*(s['full_ms']/full-1),ttft_overhead_percent=100*(s['ttft_ms']/ttft-1),
                             tpot_overhead_percent=100*(s['tpot_ms']/tpot-1),baseline_processes=len(baseline)))
    csv_write(ROOT/'analysis/e2e_latency.csv',request_rows)
    csv_write(ROOT/'analysis/scheduler_iterations.csv',iteration_rows)
    csv_write(ROOT/'analysis/gate2_sessions.csv',sessions)
    csv_write(ROOT/'analysis/gate2_p0_summary.csv',aggregate)
    csv_write(ROOT/'analysis/profiler_overhead.csv',overheads)
    resources=[];stages=[];families=[];timebreak=[];paths=[]
    for path in sorted((ROOT/'analysis').glob('gate2_*_resource_summary.csv')):
        stem=path.name.removesuffix('_resource_summary.csv');raw=load(ROOT/'raw'/(stem+'.json'));c=raw['config']
        validation=load(ROOT/'analysis'/(stem+'_validation.json'))
        assert validation['gpu_identity']=='VERIFIED_METADATA' and validation['sampling']['gap_count']==0
        for r in csv.DictReader(path.open(encoding='utf-8')):
            resources.append(dict(trace=stem,experiment=c['experiment'],precision=c['precision'],**r,
                                  timestamp_device_audit='PASSED_METADATA_AND_COVERAGE',evidence_grade='PARTIALLY_SUPPORTED_SINGLE_CAPTURE'))
    for path in sorted((ROOT/'analysis').glob('gate2_*_validation.json')):
        if path.name=='gate2_runtime_validation.json':continue
        v=load(path)
        if not isinstance(v,dict) or 'profile' not in v:continue
        if v['profile']!='P2':continue
        stem=path.name.removesuffix('_validation.json');rows=read_activity(stem);c=load(ROOT/'raw'/(stem+'.json'))['config']
        unknown=[(r['start'],r['end']) for r in rows if r['stage']=='UNKNOWN']
        all_intervals=[(r['start'],r['end']) for r in rows]
        for stage in sorted({r['stage'] for r in rows}):
            rr=[r for r in rows if r['stage']==stage]
            stages.append(dict(trace=stem,experiment=c['experiment'],precision=c['precision'],stage=stage,kernel_count=len(rr),
                               kernel_union_ms=duration([(r['start'],r['end']) for r in rr])/1e6,
                               measurement_scope='within node-level diagnostic capture; not P0 time'))
        group=defaultdict(list)
        for r in rows:group[(r['stage'],r['kernel_name'])].append(r)
        for (stage,name),rr in group.items():
            intervals=[(r['start'],r['end']) for r in rr]
            families.append(dict(trace=stem,experiment=c['experiment'],precision=c['precision'],stage=stage,kernel_name=name,
                kernel_count=len(rr),duration_sum_ms=sum(r['end']-r['start'] for r in rr)/1e6,kernel_union_ms=duration(intervals)/1e6,
                registers_per_thread=';'.join(map(str,sorted({r.get('registersPerThread','UNAVAILABLE') for r in rr},key=str))),
                static_shared_bytes=';'.join(map(str,sorted({r.get('staticSharedMemory','UNAVAILABLE') for r in rr},key=str))),
                dynamic_shared_bytes=';'.join(map(str,sorted({r.get('dynamicSharedMemory','UNAVAILABLE') for r in rr},key=str))),
                block_xyz=';'.join(sorted({str((r.get('blockX'),r.get('blockY'),r.get('blockZ'))) for r in rr})),
                logical_op='UNKNOWN',layer_id='UNKNOWN',execution_m='UNVERIFIED'))
        raw=load(ROOT/'raw'/(stem+'.json'));wall=raw['runs'][0]['full_model_wall_ms']
        info=load(ROOT/'analysis'/(stem+'_timeline.json'));window=info['cpu_full_range']
        clipped=[(max(r['start'],window['start']),min(r['end'],window['end'])) for r in rows]
        timebreak.append(dict(trace=stem,experiment=c['experiment'],precision=c['precision'],kernel_count=len(rows),
            kernel_union_ms=duration(clipped)/1e6,cpu_nvtx_window_ms=(window['end']-window['start'])/1e6,
            coverage=duration(clipped)/(window['end']-window['start']),unknown_count=len(unknown),unknown_union_ms=duration(unknown)/1e6,
            unknown_union_ratio=duration(unknown)/duration(all_intervals),profiled_model_wall_ms=wall))
        namecounts=Counter(r['kernel_name'] for r in rows)
        for tag,predicate in [('native_sm120_e2m1',lambda n:'sm120' in n.lower() and 'e2m1' in n.lower()),
                              ('gemv_family',lambda n:'gemvx' in n.lower() or 'gemv2' in n.lower()),
                              ('wmma_family',lambda n:'wmma' in n.lower()),
                              ('fp4_quantize_named',lambda n:'quant' in n.lower() and ('fp4' in n.lower() or 'e2m1' in n.lower()))]:
            selected=[r for r in rows if predicate(r['kernel_name'])]
            paths.append(dict(trace=stem,experiment=c['experiment'],precision=c['precision'],family_tag=tag,kernel_count=len(selected),
                              time_union_ms=duration([(r['start'],r['end']) for r in selected])/1e6,
                              classification='name-family evidence; not a logical-layer assignment or dynamic instruction count'))
    csv_write(ROOT/'analysis/resource_summary.csv',resources)
    if stages:csv_write(ROOT/'analysis/stage_summary.csv',stages)
    if families:csv_write(ROOT/'analysis/kernel_catalog.csv',families)
    if timebreak:csv_write(ROOT/'analysis/gate2_kernel_coverage.csv',timebreak)
    if paths:csv_write(ROOT/'analysis/gate2_kernel_paths.csv',paths)
    result=dict(p0=aggregate,overheads=overheads,kernel_coverage=timebreak,kernel_paths=paths,
                resource_scope='device-level time-weighted means; phase windows cover mapped model activity envelopes, excluding UNKNOWN activities',
                statistical_limit='Within-process CI uses five runs, t(df=4), conditional independence assumption. Across-process N is reported separately; two sessions do not support a tight population CI.')
    save(ROOT/'analysis/gate2_report_data.json',result,overwrite=True)
    print(json.dumps(dict(p0=aggregate,profiled_captures=len(overheads),node_captures=len(timebreak)),indent=2))


if __name__=='__main__':main()
