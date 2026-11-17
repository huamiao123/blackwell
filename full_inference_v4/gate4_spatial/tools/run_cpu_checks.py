"""Bound CPU test logs and structured results for all three tasks."""
from pathlib import Path
import subprocess,sys,re,json,hashlib,os
V4=Path(__file__).resolve().parents[2]
runner=Path(__file__).with_name('cpu_test_runner.py')
results=[]
temporary=V4/'local_cache/gate4_cpu_tmp'
temporary.mkdir(parents=True,exist_ok=True)
environment=dict(os.environ,TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),PYTHONIOENCODING='utf-8')
for name in ['gate4_spatial','gate4_residency','gate4_inference']:
    root=V4/name;(root/'logs').mkdir(exist_ok=True)
    command=[sys.executable,str(runner),'--tests',str(root/'tests')]
    p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=environment,timeout=90)
    log=p.stdout+p.stderr;(root/'logs/cpu_tests.log').write_text(log,encoding='utf-8',newline='\n')
    count=re.search(r'Ran (\d+) tests?',log)
    record=dict(task=name,exit_code=p.returncode,test_count=int(count[1]) if count else None,
                gpu_runtime_imports_denied=True,model_weight_reads_denied=True,
                scope='CPU correctness tests; no GPU performance or co-residency validation',
                log_sha256=hashlib.sha256((root/'logs/cpu_tests.log').read_bytes()).hexdigest())
    (root/'analysis').mkdir(exist_ok=True)
    (root/'analysis/cpu_test_results.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    results.append(record);print(name,p.returncode,record['test_count'])
    if p.returncode:print(log);raise SystemExit(p.returncode)
(V4/'gate4_spatial/analysis/combined_cpu_tests.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8',newline='\n')
