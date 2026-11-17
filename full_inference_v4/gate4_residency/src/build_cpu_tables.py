"""Sourced compiler-resource tables and theoretical models; no CUDA APIs."""
from pathlib import Path
import sys,re
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate4_spatial/src'))
from g4_common import *
from resource_model import mixed,homogeneous
from partition_model import register_partition_fit
from plan import probe_specs

ROOT=V4/'gate4_residency'

def header_audit():
    path=V4/'local_cache/gate4_cpu_sources/cuda_occupancy.h'
    if not path.exists():return None
    text=path.read_text(encoding='utf-8');names={
      'register_quantum_per_warp':'cudaOccRegAllocationGranularity',
      'shared_quantum_bytes':'cudaOccSMemAllocationGranularity',
      'subpartitions':'cudaOccSubPartitionsPerMultiprocessor'}
    out=dict(source='/usr/local/cuda-13.3/include/cuda_occupancy.h',source_sha256=sha(path),
             scope='CPU header rules for computeMajor=12; not live Occupancy API result',rules={})
    for key,name in names.items():
        at=text.index('cudaOccError '+name+'(');end=text.index('\n}',at)
        body=text[at:end];m=re.search(r'case 12:\s*value\s*=\s*(\d+)',body)
        if not m:raise ValueError('unrecognized header rule '+name)
        out[key]=int(m[1]);out['rules'][key]=dict(function=name,line=text[:at].count('\n')+1)
    out['uncertainty']='Reserved shared memory, cache carveout, dynamic register behavior and mixed scheduling not established'
    save(ROOT/'analysis/allocation_granularity_audit.json',out)
    return out

def main():
    caps=load(SPATIAL/'configs/hardware_from_gate3.json');gran=header_audit()
    compiled=read_csv(SPATIAL/'analysis/offline_compile/probe_validation.csv')
    specs=probe_specs();probes=[]
    for spec in specs:
        symbol=('fp32_' if spec['probe']=='fp32_fma' else 'tensor_')+spec['variant']
        c=next(x for x in compiled if x['symbol']==symbol)
        expected_instruction=int(c['ffma_static_instructions'] if spec['probe']=='fp32_fma' else c['bf16_mma_static_instructions'])
        if expected_instruction<=0:raise ValueError('no intended instruction confirmed for '+symbol)
        probes.append(dict(symbol=symbol,probe_type=spec['probe'],variant=spec['variant'],threads=spec['threads'],
            registers_per_thread=int(c['registers_per_thread']),static_shared_bytes=int(c['static_shared_bytes']),
            dynamic_shared_bytes=spec['shared_bytes'],actual_dynamic_shared='UNAVAILABLE_NOT_LAUNCHED',
            binary_sha256=c['cubin_sha256'],scope='offline compiler metadata + planned launch; GPU attrs/correctness not verified'))
    write_csv(ROOT/'analysis/probe_resource_table.csv',probes)
    write_csv(SPATIAL/'analysis/probe_validation.csv',compiled)
    primaries=read_csv(ROOT/'analysis/cta_resource_table.csv');matrix=[]
    for item in primaries:
        if item['logical_op'] not in ['qkv_proj','gate_up_proj']:continue
        a={k:(int(v) if k in ['threads','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','grid_ctas'] else v) for k,v in item.items()}
        for b in probes:
            for count in range(1,homogeneous(a,caps)['raw_homogeneous_cta_upper_bound']+1):
                model=mixed(a,b,caps,n_a=count,reg_quantum=gran['register_quantum_per_warp'] if gran else None,
                            shared_quantum=gran['shared_quantum_bytes'] if gran else None)
                part=register_partition_fit(dict(a,threads=a['threads']*count),b,caps,gran) if gran else dict(status='UNAVAILABLE')
                matrix.append(dict(primary_precision=a['precision'],primary_operator=a['logical_op'],primary_grid=a['grid_ctas'],
                    probe=b['symbol'],**model,register_partition_status=part['status'],model_scope='CPU theory only',
                    reserved_shared_per_cta='UNAVAILABLE_NOT_ASSUMED',partition_evidence='CUDA header model, not actual mixed scheduling'))
    write_csv(ROOT/'analysis/co_residency_matrix.csv',matrix)
    print('Generated',len(matrix),'theoretical combinations; no Occupancy or GPU calls')

if __name__=='__main__':main()
