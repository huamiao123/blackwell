"""Describe actual Decode node durations in each historic full-inference Node trace."""
import csv
import gzip
import json
from pathlib import Path
from statistics import median, quantiles
from analyze import ROOT, write

V4=ROOT.parent
def main():
    origin=V4/'gate4_spatial/qkv_replay_validation/analysis/tensor64_projection_comparison_20261119.csv'
    with origin.open() as f:
        pilots={(r['precision'],r['operator']):float(r['p2_b_alone_ns']) for r in csv.DictReader(f)
                if r['order']=='ab'}
    results=[]
    for precision,folder in [('bf16','gate3_backbone_bf16'),('nvfp4','gate3_backbone')]:
        with gzip.open(V4/folder/'analysis/attributed_activities.json.gz','rt') as f:
            events=json.load(f)
        groups={}
        for item in events:
            if item.get('operator_evidence')!='VERIFIED_EXACT_WEIGHT_REFERENCE_AND_CUDA_TOOLS_ID':
                continue
            if item['stage']!='DECODE': raise ValueError('unexpected stage')
            op=item['logical_op']
            duration=item['end']-item['start']
            if duration<=0: raise ValueError('nonpositive kernel')
            for layer in ('all',str(item['layer_id'])):
                groups.setdefault((op,layer),[]).append(duration)
        del events
        for (op,layer),ns in groups.items():
            if len(ns)!=(9216 if layer=='all' else 256):
                raise ValueError('incomplete node set')
            threshold=pilots[precision,op]
            p=quantiles(ns,n=100,method='inclusive')
            results.append(dict(precision=precision,operator=op,layer=layer,kernels=len(ns),
                                p05_us=p[4]/1000,p50_us=median(ns)/1000,p95_us=p[94]/1000,
                                isolated_b_alone_us=threshold/1000,
                                fraction_duration_ge_b_alone=sum(n>=threshold for n in ns)/len(ns),
                                scope='P2 real Decode node span; isolated B reference, no injected work'))
    write(ROOT/'analysis/full_decode_duration_bounds.csv',results)
if __name__=='__main__':
    main()
