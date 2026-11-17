"""CPU review-package provenance and absence-of-measurement checks."""
from pathlib import Path
import argparse,csv,hashlib,json,subprocess,sys

V4=Path(__file__).resolve().parents[2]
REPO=V4.parent
TASKS=['gate4_spatial','gate4_residency','gate4_inference']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def validate(write_manifest=False):
    historical=[]
    for rel in ['analysis/gate2_artifact_manifest.json','gate3_attribution/analysis/artifact_manifest.json',
                'gate3_backbone/analysis/artifact_manifest.json']:
        artifacts=load(V4/rel)['artifacts']
        for item in artifacts:
            p=V4/item['path']
            if sha(p)!=item['sha256']:raise ValueError('historical artifact changed: '+item['path'])
        historical.append(dict(manifest=rel,count=len(artifacts),all_hashes_unchanged=True))
    tests=[load(V4/t/'analysis/cpu_test_results.json') for t in TASKS]
    if any(t['exit_code'] or not t['gpu_runtime_imports_denied'] or not t['model_weight_reads_denied'] for t in tests):
        raise ValueError('CPU checks not passed under runtime/weight guards')
    if sum(t['test_count'] for t in tests)!=25:raise ValueError('unexpected test inventory; review report counts')
    for t in tests:
        if sha(V4/t['task']/'logs/cpu_tests.log')!=t['log_sha256']:raise ValueError('test log hash mismatch')
    audit=load(V4/'gate4_spatial/analysis/input_audit.json')
    for record in audit['files']:
        if sha(V4/record['path'])!=record['sha256']:raise ValueError('audited input changed: '+record['path'])
    if audit['gpu_started'] or audit['gate4_valid_gpu_evidence_present']:raise ValueError('CPU release cannot include new GPU evidence')
    primary=rows(V4/'gate4_residency/analysis/cta_resource_table.csv')
    if len(primary)!=8:raise ValueError('primary inventory incomplete')
    models=rows(V4/'gate4_residency/analysis/co_residency_matrix.csv')
    if len(models)!=56 or any(r['actual_kernel_overlap']!='UNAVAILABLE_NO_GPU_RUN' for r in models):
        raise ValueError('resource model scope/count incorrect')
    compiled=load(V4/'gate4_spatial/analysis/offline_compile/compile_manifest.json')
    if not compiled['cpu_only'] or compiled['gpu_module_loaded']:raise ValueError('not offline-only compilation')
    if sha(V4/'gate4_spatial/src/probes.cu')!=compiled['source_sha256']:raise ValueError('CUDA source changed since offline compile')
    for row in rows(V4/'gate4_spatial/analysis/probe_validation.csv'):
        if row['cubin_sha256']!=compiled['cubin_sha256'] or row['gpu_correctness']!='UNVERIFIED':
            raise ValueError('mislabelled compiled probe evidence')
        if row['sass_sha256']!=sha(V4/'gate4_spatial/analysis/offline_compile/sass.txt'):
            raise ValueError('SASS hash mismatch')
    empty={
      'gate4_spatial':['probe_standalone','concurrency_runs','concurrency_summary','spatial_concurrency','interference_matrix'],
      'gate4_residency':['concurrency_results','interference_results'],
      'gate4_inference':['prefill_concurrency','decode_concurrency','request_metrics','auxiliary_work','concurrency_summary',
                         'e2e_comparison','resource_aware_policy','paired_runs']}
    for task,names in empty.items():
        for name in names:
            if rows(V4/task/'analysis'/f'{name}.csv'):raise ValueError('unexpected GPU observations in CPU release')
    for task in TASKS:
        root=V4/task;c=load(root/'configs/review.json')
        if c['gpu_authorized'] or c['auto_next_gate'] or c['idle_polling'] or c['review_status']!='CPU_REVIEW_PENDING':
            raise ValueError('unsafe review defaults')
        if load(root/'configs/gpu_window.example.json')['enabled']:raise ValueError('active example window')
        for name in ['README.md','methodology.md','final_report.md']:
            if not (root/name).read_text(encoding='utf-8').strip():raise ValueError('missing review document')
        # Parse without imports: not a CUDA runtime or a model integration test.
        for p in [*(root/'src').glob('*.py'),*(root/'tests').glob('*.py'),*(root/'tools').glob('*.py')]:
            compile(p.read_text(encoding='utf-8'),str(p),'exec')
    output=dict(status='CPU_PACKAGE_VALIDATED',new_gpu_workloads=0,model_weights_loaded=False,
                historical_manifests=historical,total_cpu_tests=25,compiled_probe_count=4,
                theoretical_combinations=56,new_gpu_observation_rows=0,
                limitations=['live activation replay adapter not validated','no live Occupancy API',
                             'no GPU probe output test','no useful consumer or E2E gain demonstrated'])
    out=V4/'gate4_spatial/analysis/package_validation.json'
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    if write_manifest:
        files=[]
        for task in TASKS:
            for p in sorted((V4/task).rglob('*')):
                if not p.is_file() or any(x in p.relative_to(V4/task).parts for x in ['__pycache__','build']):continue
                if p.name in ['artifact_manifest.json','gpu_window.json'] or p.suffix in ['.pyc','.nsys-rep','.sqlite','.cubin']:continue
                files.append(dict(path=p.relative_to(V4).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
        p=V4/'gate4_cpu_review.md';files.append(dict(path=p.relative_to(V4).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
        release=dict(scope='CPU preparation only; no new GPU experiments',model_weights_loaded=False,
                     historical_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                     artifacts=files)
        (V4/'gate4_spatial/analysis/artifact_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-manifest',action='store_true')
    validate(p.parse_args().write_manifest)
