"""Join isolated diagnostics to archived P0/P2 evidence by precision and operator."""
from collections import defaultdict
import csv
from statistics import median
from analyze import ROOT, load, write

V4=ROOT.parent
PILOTS=V4/'gate4_spatial/qkv_replay_validation/analysis'
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def main():
    summaries=rows(ROOT/'analysis/summary.csv')
    cases={(r['precision'],r['operator']) for r in summaries}
    profile={}
    for path in sorted((ROOT/'raw').glob('*/identity.json')):
        precision,op,_,_=load(path.parent/'case.json')['case']
        profile[precision,op]=load(path)
    windows={(r['precision'],r['operator']):r for r in rows(ROOT/'analysis/full_decode_duration_bounds.csv') if r['layer']=='all'}
    p2={(r['precision'],r['logical_op']):r for r in rows(V4/'gate3_backbone/analysis/precision_comparison.csv')}
    history=defaultdict(list)
    for r in rows(PILOTS/'tensor64_projection_comparison_20261119.csv'):
        history[r['precision'],r['operator']].append(r)
    fns={'qkv_proj':'qkv_grid64_fixed','gate_up_proj':'gate_up_grid64','down_proj':'down_grid64','o_proj':'o_grid64'}
    result=[]
    for p,op in sorted(cases):
        s=[r for r in summaries if (r['precision'],r['operator'])==(p,op)]
        def metric(key,cache='none',group='resources'):
            good=[r for r in s if r['counter_set']==group and r['cache_control']==cache and r['metric']==key and r['median']]
            return float(good[0]['median']) if good else None
        a=profile[p,op]
        m,k=a['input_shape']
        if a['output_shape'][0]!=m: raise ValueError('invalid output shape')
        n=a['output_shape'][1]
        live=a['live_identity']
        p0=[float(r['a_ns']) for r in rows(PILOTS/f'tensor_light_{fns[op]}_p0_conditions.csv')
            if r['mode']=='a_alone' and r['precision']==p and r['correctness_passed']=='True']
        run=windows[p,op]
        probe=history[p,op]
        if len(p0)!=6 or len(probe)!=2: raise ValueError('incomplete paired reference')
        row=dict(precision=p,operator=op,m=m,n=n,k=k,grid_ctas=p2[p,op]['grid_ctas'],
                 regs_per_thread=live['register_and_shared_attributes']['NUM_REGS'],
                 dynamic_smem_bytes=live['dynamic_shared_bytes'],api_blocks_per_sm=live['homogeneous_api_blocks_per_sm'],
                 isolated_p0_event_us=median(p0)/1000,isolated_p0_effective_logical_tflop_s=2*m*n*k/(median(p0)*1000),
                 isolated_p3_us=metric('gpu__time_duration.sum')/1000,
                 full_decode_p2_kernel_median_us=run['p50_us'],full_decode_p2_fraction_ge_b=run['fraction_duration_ge_b_alone'],
                 share_of_full_p2=p2[p,op]['share_of_own_p2_full_window'],
                 p0_b_retention_ab=next(r['p0_probe_throughput_retention_mean'] for r in probe if r['order']=='ab'),
                 p0_b_retention_ba=next(r['p0_probe_throughput_retention_mean'] for r in probe if r['order']=='ba'),
                 p3_tensor_elapsed_pct=metric('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
                 p3_fma_elapsed_pct=metric('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed'),
                 p3_sm_active_elapsed_pct=metric('sm__cycles_active.avg.pct_of_peak_sustained_elapsed',group='stalls'),
                 p3_tensor_active_denominator_pct=metric('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',group='stalls'),
                 p3_eligible_warps=metric('smsp__warps_eligible.avg.per_cycle_active'),
                 p3_long_scoreboard_pct=metric('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),
                 p3_dram_read_bytes=metric('dram__bytes_op_read.sum'),
                 p3_flush_dram_read_bytes=metric('dram__bytes_op_read.sum',cache='all'))
        result.append(row)
    write(ROOT/'analysis/projection_comparison.csv',result)
if __name__=='__main__':
    main()
