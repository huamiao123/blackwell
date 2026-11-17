"""Only apply explicit, trace-specific reviewed operator mappings. Default UNKNOWN."""
import argparse
from common import load,save,csv_write


def map_rows(rows,contract,trace_sha256):
    if contract and contract['trace_sha256']!=trace_sha256:raise ValueError('mapping belongs to another trace')
    entries=(contract or {}).get('entries',[]);out=[]
    for r in rows:
        matches=[]
        for m in entries:
            keys=['contextId','graphId','graphNodeId'] if m.get('identity_kind')=='graph_node' else ['contextId','correlationId']
            if any(m.get(k) is None or r.get(k) is None for k in keys):continue
            if all(str(m[k])==str(r[k]) for k in keys) and m['kernel_name']==r['kernel_name']:matches.append(m)
        mapped=dict(r,logical_op='UNKNOWN',layer_id='UNKNOWN',logical_m='UNVERIFIED',execution_m='UNVERIFIED',mapping_status='UNVERIFIED')
        if len(matches)==1:
            m=matches[0]
            if m.get('review_status')=='VERIFIED' and m.get('source_evidence') and m.get('shape_evidence'):
                mapped.update({k:m[k] for k in ['logical_op','layer_id','logical_m','execution_m']})
                mapped['mapping_status']='VERIFIED_REVIEWED_TRACE_CONTRACT'
        out.append(mapped)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activities',required=True);p.add_argument('--trace-sha256',required=True)
    p.add_argument('--contract');p.add_argument('--output',required=True);a=p.parse_args()
    rows=map_rows(load(a.activities),load(a.contract) if a.contract else None,a.trace_sha256)
    save(a.output,rows,overwrite=True)
    from pathlib import Path
    fields=['stage','layer_id','logical_op','kernel_name','logical_m','execution_m','mapping_status','contextId','graphId','graphNodeId','correlationId']
    csv_write(Path(a.output).with_suffix('.csv'),[{k:r.get(k,'UNAVAILABLE') for k in fields} for r in rows],fields)
