"""CPU audit manifest and honest tool-availability inventory; no hardware sampling."""
import csv
from common import ROOT,load,save,file_hash,csv_write

FILES=['nsys_round2/final_report.md','nsys_round2/analysis/metric_summary.csv',
       'nsys_round2/analysis/profiler_overhead.csv','nsys_round2/src/original/decode_round2.py',
       'resource_overlap_v1/final_report.md','resource_overlap_v1/methodology.md',
       'resource_overlap_v1/src/overlap_runner.py','resource_overlap_v1/src/analyze_overlap.py',
       'resource_overlap_v1/analysis/failures.md','audit_v3/corrected_results.md','audit_v3/source_audit.md']


def run():
    files=[]
    for name in FILES:
        p=ROOT.parent/name;raw=p.read_bytes();text=raw.decode('utf-8-sig')
        files.append(dict(path=name,sha256=file_hash(p),bytes=len(raw),lines=len(text.splitlines())))
    save(ROOT/'analysis/history_sources.json',files,overwrite=True)
    old=list(csv.DictReader((ROOT.parent/'nsys_round2/analysis/metric_summary.csv').open(encoding='utf-8-sig')))
    names=sorted({str(v) for r in old for k,v in r.items() if k in ['name','metric','metric_name','metricName'] and v})
    rows=[]
    for name in names:
        rows.append(dict(metric=name,tool='NSYS',availability='HISTORICALLY_OBSERVED',measurement_scope='old device-level trace',
                         measurement_status='UNVERIFIED_V41_NO_COLLECTION',evidence=FILES[1],unit='preserve original exported metric name/unit'))
    for name in ['FP32/FMA pipe','INT/ALU pipe','L2 hit rate','L1/TEX hit rate','Achieved occupancy','Eligible warps',
                 'Warp stall reasons','Dynamic MMA instruction count','Shared-memory bank conflicts','Register spills']:
        rows.append(dict(metric=name,tool='NCU',availability='UNVERIFIED_INJECTION',measurement_scope='target kernel if collection succeeds',
                         measurement_status='UNAVAILABLE_CURRENT_EVIDENCE',evidence='historical vLLM injection failure; current help/version only',unit='UNVERIFIED'))
    for name in ['Grid/block','Registers per thread','Static/dynamic shared memory','Context/stream/correlation/Graph IDs']:
        rows.append(dict(metric=name,tool='NSYS launch metadata',availability='SCHEMA_DEPENDENT',measurement_scope='recorded launch, not utilization',
                         measurement_status='UNVERIFIED_NEW_TRACE',evidence='Gate 2 PRAGMA schema check required',unit='threads/registers/bytes/identifiers'))
    csv_write(ROOT/'analysis/metric_availability.csv',rows)
    contract=load(ROOT/'prompts/manifests/tokenizer_contract.json')
    env=load(ROOT/'analysis/cpu_environment_audit.json')
    assets=env['tokenizer_asset_hashes']
    same=assets['Qwen3-8B-BF16']['tokenizer.json']==assets['Qwen3-8B-NVFP4']['tokenizer.json']
    save(ROOT/'analysis/tokenizer_asset_audit.json',dict(defined_ids=contract['defined_id_count'],
         tokenizer_json_hash_identical=same,config_hash_identical=assets['Qwen3-8B-BF16']['tokenizer_config.json']==assets['Qwen3-8B-NVFP4']['tokenizer_config.json'],
         canonical_policy='All inputs rendered/encoded with BF16 tokenizer, shared identical IDs; thinking=False.',
         assets=assets,runtime_stop_policy='fixed length: ignore_eos=True; natural EOS deferred explicit normalization',gpu_started=False),overwrite=True)
    schemas={
      'e2e_latency':['precision','experiment','profile','trial','request_id','ttft_ms','tpot_ms','itl_p95_ms','model_request_ms','measurement_scope','measurement_status'],
      'stage_summary':['precision','stage','activity_union_ns','measurement_scope','measurement_status'],
      'scheduler_iterations':['precision','request_id','scheduler_iteration','scheduled_prefill_tokens','scheduled_decode_tokens','active_request_count','measurement_status'],
      'kernel_catalog':['precision','stage','layer_id','logical_op','kernel_family','kernel_name','kernel_duration','kernel_count','grid_size','block_size','contextId','streamId','graphId','graphNodeId','correlationId','measurement_status'],
      'operator_mapping':['precision','stage','layer_id','logical_op','kernel_name','mapping_status','source_evidence','shape_evidence'],
      'operator_timing':['precision','stage','layer_id','logical_op','kernel_union_ns','mapping_status'],
      'resource_summary':['precision','stage','sm_active','sm_issue','tensor_active','fp32_pipe','int_pipe','dram_read','dram_write','l2_throughput','l2_hit_rate','registers','shared_memory','occupancy','eligible_warps','stall_reason','clock','measurement_tool','measurement_scope','measurement_status','confidence'],
      'profiler_overhead':['config_hash','profile','baseline_mean_ms','profiled_mean_ms','overhead_ratio','measurement_status'],
      'interference_matrix':['precision','condition','probe','pressure_level','ia','rb','s_ab','s_ba','s_fastest_serial','b_work_hash','a_work_hash','measurement_status'],
      'paired_runs':['block_id','order','mode','a_ms','b_span_ms','joint_ms','a_work_hash','b_work_hash','measurement_status']}
    for name,fields in schemas.items():
        path=ROOT/'analysis'/(name+'.csv')
        if not path.exists():csv_write(path,[],fields)
    print('AUDITED',len(files),'historical files;',len(names),'historically observed NSYS metric labels;',len(rows),'availability rows')


if __name__=='__main__':run()
