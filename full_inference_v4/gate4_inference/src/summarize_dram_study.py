"""Audit E2 streamed bytes, outputs, stage-local timelines and paired request timing."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
import sys
from analyze_gate44_p0 import analyze, request_itl
from common import csv_write, percentile


def process(raw,out):
    pristine={}
    standalone=[]
    inventory=[]
    failed=[]
    p0=[[],[],[]]
    timelines={}
    full_metrics={}
    active_window=[]
    for directory in sorted(raw.glob('dram44_*')):
        rid=directory.name
        guard=json.loads((directory/(rid+'_supervision.json')).read_bytes())
        if guard['status']!='COMPLETED':
            failed.append(dict(run_id=rid,status=guard['status'],cause=guard['error'],source_log=rid+'.log'))
            continue
        with gzip.open(directory/(rid+'.json.gz'),'rt') as f:
            run=json.load(f)
        if 'results' in run:
            for row in run['results']:
                if not row['correctness_passed'] or row['region_bytes']!=4*run['l2_bytes']:
                    raise ValueError('failed standalone correctness')
                standalone.append(dict(run_id=rid, profile=run['profile'],gpu_uuid=run['gpu_uuid'],
                                       logical_total_GBps=(row['logical_read_bytes']+row['logical_write_bytes'])/row['median_us']/1000,
                                       **{k:v for k,v in row.items() if k!='gpu_us'}))
            continue
        if run['invocation_status']!='COMPLETED':
            raise ValueError('incorrect model status')
        work=run['gate44']['probe']
        precision=run['config']['precision']
        cond=run['gate44']['condition']
        if (work['symbol'],work['grid'],work['logical_read_bytes'],work['logical_write_bytes'])!=('dram_read_light',128,402653184,49152):
            raise ValueError('E2 probe work mismatch')
        profile=run['profile']
        for model,aux in zip(run['runs'],run['gate44']['trials'],strict=True):
            hashes=tuple(sorted((r['prompt_hash'],r['output_hash']) for r in model['request_summary']))
            identity=(hashes,run['config_file_sha256'],run['prompt_manifest_hash'],run['gpu_uuid'],run['gate44']['cubin_sha256'])
            if pristine.setdefault(precision,identity)!=identity or len({h[0] for h in hashes})!=8:
                raise ValueError('input/output/configuration mismatch')
            expected=32 if aux['condition'] in ('parallel','serial') else 0
            if (len(model['graph_replays']),model['total_output_tokens'],aux['completed_probes'])!=(256,2056,expected):
                raise ValueError('incorrect work')
            if expected and (len(aux['completions'])!=32 or any(
                    not r['correctness_passed'] or r['checked_uint32']!=work['tiles'] for r in aux['completions'])):
                raise ValueError('probe verification incomplete')
            if profile=='P0':
                groups=[request_itl(r)[32:64] for r in model['requests'].values()]
                gaps=[ms for group in groups for ms in group]
                active_window.append(dict(run_id=rid,precision=precision,block=model['trial']//4,condition=aux['condition'],
                                          output_gap_mean_ms=mean(gaps),output_gap_p95_ms=percentile(gaps,.95),
                                          logical_read_bytes=expected*work['logical_read_bytes']))
        inventory.append(dict(run_id=rid,precision=precision,profile=profile,condition=cond,request_groups=len(run['runs']),
                              uuid=run['gpu_uuid'],config_sha256=run['config_file_sha256'],prompt_sha256=run['prompt_manifest_hash'],
                              binary_sha256=run['gate44']['cubin_sha256'],checked_outputs=True))
        if profile=='P0':
            for target,rows in zip(p0,analyze(run,guard)):
                target.extend(rows)
        elif profile=='P2':
            trace=json.loads((directory/'verified_intersections.json').read_bytes())
            if trace['verified_targets']!=32 or trace['overlap_count']!=(32 if cond=='parallel' else 0):
                raise ValueError('unexpected GPU overlap')
            timelines[precision,cond]=trace
        elif profile=='P1':
            full_metrics[precision,cond]=json.loads((out/'decode'/(rid+'.json')).read_bytes())
    csv_write(out/'standalone.csv',standalone)
    csv_write(out/'run_inventory.csv',inventory)
    if failed: csv_write(out/'failed_runs.csv',failed)
    for rows,name in zip(p0,('request_envelopes','paired_blocks','statistics')):
        if rows:
            csv_write(out/(name+'.csv'),rows)
    if active_window:
        csv_write(out/'target_output_intervals.csv',active_window)
    contrasts=[]
    blocks=defaultdict(dict)
    for row in active_window:
        blocks[row['run_id'],row['precision'],row['block']][row['condition']]=row
    estimates=defaultdict(list)
    for (rid,precision,_),cases in blocks.items():
        for feature in ('output_gap_mean_ms','output_gap_p95_ms'):
            estimates[rid,precision,feature].append(cases['parallel'][feature]/cases['a_alone'][feature]-1)
    for (rid,precision,feature),values in estimates.items():
        if len(values)!=5:raise ValueError('incomplete block set')
        m=mean(values)
        h=2.776445*stdev(values)/5**.5
        contrasts.append(dict(run_id=rid,precision=precision,metric=feature,paired_mean_change=m,ci95_low=m-h,ci95_high=m+h))
    if contrasts:
        csv_write(out/'target_latency_change.csv',contrasts)
    result=[]
    conditional=[]
    for precision in ('nvfp4','bf16'):
        a,b=(timelines[precision,k] for k in ('parallel','serial'))
        arows,brows=a['rows'],b['rows']
        if [(r['iteration'],r['primary_kernels']) for r in arows]!=[(r['iteration'],r['primary_kernels']) for r in brows]:
            raise ValueError('primary graph changed')
        d=lambda rs:[(r['probe_end_ns']-r['probe_start_ns'])/1000 for r in rs]
        p,s=d(arows),d(brows)
        gp=lambda rs:mean((r['primary_last_ns']-r['primary_first_ns'])/1000 for r in rs)
        profiles=[('P2',p,s)]
        if all((precision,k) in full_metrics for k in ('parallel','serial')):
            profiles.append(('P1',full_metrics[precision,'parallel']['probe_kernel_us'],full_metrics[precision,'serial']['probe_kernel_us']))
        for profile,pus,sus in profiles:
            result.append(dict(precision=precision,profile=profile,parallel_us=median(pus),serial_us=median(sus),
                               conditional_retention=sum(sus)/sum(pus),parallel_logical_read_GBps=402653184/(mean(pus)*1000),
                               serial_logical_read_GBps=402653184/(mean(sus)*1000),
                               parallel_target_graph_us=gp(arows) if profile=='P2' else 'NOT_APPLICABLE',
                               serial_target_graph_us=gp(brows) if profile=='P2' else 'NOT_APPLICABLE',
                               fraction_probe_covered=sum(r['overlap_ns'] for r in arows)/(sum(p)*1000) if profile=='P2' else 'NOT_APPLICABLE'))
        for cond in ('parallel','serial'):
            if (precision,cond) not in full_metrics:
                continue
            obj=full_metrics[precision,cond]
            for row in obj['conditional_device_metrics']:
                conditional.append(dict(precision=precision,condition=cond,run_id=obj['run_id'],**row))
    csv_write(out/'probe_competition.csv',result)
    if conditional:
        csv_write(out/'device_windows.csv',conditional)


if __name__=='__main__':
    process(Path(sys.argv[1]),Path(sys.argv[2]))
