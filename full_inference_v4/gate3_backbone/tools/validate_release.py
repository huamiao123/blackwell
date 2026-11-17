"""CPU integrity audit of exported capture metadata and exact module references."""
from pathlib import Path
import csv,gzip,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
V4=ROOT.parent


def main():
    checks=[]
    for folder,stem in [('gate3_backbone','gate3_bb_e2_nvfp4_p2_r01'),
                        ('gate3_backbone_bf16','gate3_bb_e2_bf16_p2_r02')]:
        path=V4/'raw'/(stem+'.json.gz')
        with gzip.open(path,'rb') as f:b=f.read()
        raw=json.loads(b)
        manifest=json.loads((V4/folder/'analysis/raw_trace_manifest.json').read_text(encoding='utf-8'))
        expected=next(r['sha256'] for r in manifest if r['path']=='raw/'+stem+'.json')
        assert hashlib.sha256(b).hexdigest()==expected
        assert raw['invocation_status']=='COMPLETED'
        for filename,digest in raw['code_sha256'].items():assert hashlib.sha256((V4/'src'/filename).read_bytes()).hexdigest()==digest
        c=raw['config'];assert c['batch']==8 and not c['engine']['enable_prefix_caching']
        assert not c['engine']['enforce_eager'] and c['engine']['kv_cache_dtype']=='fp8_e4m3'
        assert hashlib.sha256((V4/'configs'/('E2_'+c['precision']+'.json')).read_bytes()).hexdigest()==raw['config_file_sha256']
        meta=raw['backbone_metadata'];used={r['capture_index'] for r in meta['replays']};assert len(used)==1
        capture=meta['captures'][next(iter(used))]
        nodes=capture['native_kernels' if c['precision']=='nvfp4' else 'matrix_kernels']
        assert len(nodes)==144 and len(meta['replays'])==256
        weights={t['module']:t for t in meta['tensor_catalog'] if t['tensor']=='weight'}
        modules=set()
        for n in nodes:
            payload=bytes.fromhex(n['parameter_bytes_hex'])
            assert len(payload)==n['parameter_size']
            assert hashlib.sha256(payload).hexdigest()==n['parameter_sha256']
            name=n['binding']['module'];modules.add(name)
            assert weights[name]['address'].to_bytes(8,'little') in payload
        assert len(modules)==144
        rows=list(csv.DictReader((V4/folder/'analysis/layer_operator_timing.csv').open(encoding='utf-8')))
        assert len(rows)==144 and all(int(r['kernel_count'])==256 for r in rows)
        result=json.loads((V4/folder/'analysis/validation.json').read_text(encoding='utf-8'))
        assert result['matched_decode_kernel_executions']==36864 and result['same_kernel_name_multiset_as_gate2']
        checks.append(dict(precision=c['precision'],original_json_sha256=expected,modules=144,kernel_executions=36864))
    out=dict(status='PASSED_CPU_EXPORT_AND_REFERENCE_INTEGRITY',gpu_launched=False,cases=checks,
             limits='SQL/CUPTI joins were validated on the server; this release check verifies exported metadata and byte integrity, not new GPU measurements')
    (ROOT/'analysis/release_validation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS: both compressed originals match server hashes; 288 module references and 73728 executions audited')


if __name__=='__main__':main()
