"""Final CPU-only QA. No GPU query, model import, inference or profiler target."""
import ast
import datetime
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT,file_hash,load,save


def main():
    (ROOT/'local_cache').mkdir(exist_ok=True)
    source=[]
    for f in [*(ROOT/'src').glob('*.py'),*(ROOT/'tests').glob('*.py')]:
        ast.parse(f.read_text(encoding='utf-8'));source.append(f.relative_to(ROOT).as_posix())
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],capture_output=True,text=True,timeout=90)
    (ROOT/'analysis/cpu_tests.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    if list((ROOT/'raw').glob('*')) if (ROOT/'raw').exists() else False:
        raise RuntimeError('Gate 0/1 snapshot unexpectedly has raw results')
    required=['implementation_report.md','history_audit.md','research_hypotheses.md','prompt_protocol.md',
              'metric_inventory.md','timing_protocol.md','concurrency_protocol.md']
    for name in required:
        if not (ROOT/name).is_file():raise ValueError('missing '+name)
    p=load(ROOT/'analysis/prompt_validation.json')
    if p['prompts']!=88 or len(p['configs'])!=14:raise ValueError('dataset completeness')
    if len(p.get('real_tokenizer_negative_tests_passed',[]))!=5:raise ValueError('tokenizer negative tests missing')
    w=load(ROOT/'configs/gpu_window.example.json')
    if w['enabled'] or w['reviewed_gates']:raise ValueError('GPU gate unexpectedly enabled')
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),scope='Gate 0/1 CPU preparation',
                gpu_workloads_started=0,formal_model_inferences=0,new_ncu_injections=0,
                cpu_tests_status='VERIFIED',cpu_tests_log='analysis/cpu_tests.log',
                actual_tokenizer_prompts=88,matched_configurations=14,real_tokenizer_negative_tests=5,
                ast_checked=source,gpu_runtime_integration='UNVERIFIED',kernel_mapping_on_new_trace='UNVERIFIED',
                teacher_forcing='DESIGN_ONLY',probe_pressure_and_stage_injection='DESIGN_ONLY',
                new_resource_results='UNVERIFIED',gate2_authorization='DISABLED_PENDING_MANUAL_REVIEW',
                implementation_code_sha256={f.name:file_hash(f) for f in (ROOT/'src').glob('*.py')})
    save(ROOT/'analysis/quality_validation.json',result,overwrite=True)
    print(json.dumps({k:v for k,v in result.items() if k!='implementation_code_sha256'},indent=2))


if __name__=='__main__':main()
