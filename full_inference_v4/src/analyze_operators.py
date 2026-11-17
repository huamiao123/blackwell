"""Union timings for reviewed mappings; fused kernels counted once."""
from collections import defaultdict
from common import duration


def operator_timings(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['activity_kind']!='KERNEL':continue
        key=(r['stage'],str(r['layer_id']),r['logical_op'],r['mapping_status'])
        groups[key].append((r['start'],r['end']))
    return [dict(stage=k[0],layer_id=k[1],logical_op=k[2],mapping_status=k[3],
                 kernel_count=len(v),kernel_union_ns=duration(v),measurement_scope='GPU operator active-time union')
            for k,v in groups.items()]


if __name__=='__main__':
    import argparse
    from common import load,csv_write
    p=argparse.ArgumentParser();p.add_argument('--mapped-activities',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    csv_write(a.output,operator_timings(load(a.mapped_activities)))
