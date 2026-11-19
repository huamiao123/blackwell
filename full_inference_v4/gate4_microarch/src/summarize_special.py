"""Audit sampled PCs and single-pass L2 read-lookup sector counts."""
from analyze import ROOT,load,read_run,write
def main():
    pcs=[]
    probes=[]
    for path in sorted((ROOT/'raw').iterdir()):
        if not (path/'profiled_symbols.json').exists():continue
        precision,op,group,cache=load(path/'case.json')['case']
        if group not in ('pc','cache_read'):continue
        vals=read_run(path)
        if group=='pc':
            data=load(path/'pc_samples.json')
            total=data['smsp__pcsamp_sample_count']
            sleep=data['smsp__pcsamp_warps_issue_stalled_sleeping']
            if sum(i['samples'] for i in total['instances'])!=total['total']:raise ValueError('missing sample addresses')
            if sum(i['samples'] for i in sleep['instances'])!=sleep['total']:raise ValueError('sleep samples do not close')
            sites=sleep['instances']
            pairs=sum(i['samples'] for i in sites if 'SYNCS.PHASECHK' in i['sass'] and 'NANOSLEEP.SYNCS' in i['previous_sass'])
            top=max((i['samples'] for i in sites),default=0)
            pcs.append(dict(run_id=path.name,precision=precision,operator=op,all_pc_samples=total['total'],
                            sleeping_samples=sleep['total'],sleeping_pct=100*sleep['total']/total['total'] if total['total'] else None,
                            sleep_pc_sites=len(sites),sleep_at_phasecheck_after_nanosleep=pairs,
                            fraction_of_sleep_at_pair=pairs/sleep['total'] if sleep['total'] else None,
                            samples_at_hottest_sleep_pc=top,passes=vals[0]['passes']))
        else:
            if vals[0]['passes']!=1:status='MULTIPASS'
            else:status='ONE_PASS'
            m={r['metric']:int(r['value']) for r in vals}
            h=m['lts__t_sectors_op_read_lookup_hit.sum']
            miss=m['lts__t_sectors_op_read_lookup_miss.sum']
            probes.append(dict(run_id=path.name,precision=precision,operator=op,cache_control=cache,passes=vals[0]['passes'],
                               hit_read_sectors=h,miss_read_sectors=miss,read_lookup_hit_pct=100*h/(h+miss) if h+miss else None,
                               scope='L2 read lookup sectors',evidence=status))
    if pcs:write(ROOT/'analysis/pc_waits.csv',pcs)
    if probes:write(ROOT/'analysis/l2_read_lookups.csv',probes)
if __name__=='__main__':
    main()
