"""CPU integrity/fairness checks and a publishable SHA256 manifest."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    for f in (root/'src').rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8'))
    for record in json.loads((root/'analysis/measured_source_map.json').read_text()):
        assert hashlib.sha256((root/record['matching_source']).read_bytes()).hexdigest()==record['recorded_sha256']
    for script in ['test_analysis_cpu.py','summarize_baseline.py']:
        r=subprocess.run([sys.executable,str(root/'src'/script)],capture_output=True,text=True)
        if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    data=json.loads((root/'analysis/baseline_validation.json').read_text())
    assert data['valid_trials']==20
    assert json.loads((root/'analysis/timeline_validation.json').read_text())==[], 'Update verification for a later completed GPU phase'
    files=[]
    for f in sorted(root.rglob('*')):
        if not f.is_file() or any(x in f.parts for x in ['__pycache__','local_cache']):continue
        if f.name=='bundle_manifest.json':continue
        assert f.suffix not in ['.cubin','.nsys-rep','.sqlite'], ('binary excluded from publication',str(f))
        b=f.read_bytes()
        files.append(dict(path=f.relative_to(root).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    (root/'analysis/bundle_manifest.json').write_text(json.dumps(files,indent=2))
    print('VERIFIED',len(files),'files; 20 fair trials; measured-source hashes exact; no new overlap claim')


if __name__=='__main__':main()
