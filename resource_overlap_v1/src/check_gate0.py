"""CPU-only evidence and protocol checks. Does not import torch or Triton."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy


def main():
    out = Path(__file__).resolve().parents[1]
    root = out.parent
    required = ['nsys_round2/final_report.md', 'nsys_round2/baseline.csv',
                'nsys_round2/analysis/metric_summary.csv', 'nsys_round2/analysis/profiler_overhead.csv',
                'nsys_round2/src/original/decode_round2.py', 'nsys_round2/src/original/analyze_round2.py',
                'audit_v3/corrected_results.md', 'audit_v3/raw/gate9_M1024.json', 'audit_v3/source_audit.md']
    inputs = []
    for name in required:
        raw = (root / name).read_bytes()
        text = raw.decode('utf-8-sig')
        inputs.append(dict(path=name, bytes=len(raw), lines=len(text.splitlines()), sha256=hashlib.sha256(raw).hexdigest()))
    code = []
    for f in sorted((out / 'src').glob('*.py')):
        ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
        code.append(f.name)
    prompts = runpy.run_path(str(out / 'src/overlap_runner.py'))['prompt_ids']
    ids = prompts(8)
    assert len(ids) == 8 and all(len(x) == 2048 for x in ids)
    assert len({tuple(x) for x in ids}) == 8 and prompts(1)[0] == ids[0]
    assert all(0 <= i < 151936 for x in ids for i in x)
    baseline = list(csv.DictReader(io.StringIO((root / required[1]).read_text(encoding='utf-8-sig'))))
    audit = json.loads((root / required[7]).read_text(encoding='utf-8-sig'))
    result = dict(status='CPU protocol checks passed; GPU correctness untested', gpu_work_launched=False,
                  inputs=inputs, python_ast_checked=code, old_baseline_rows=len(baseline),
                  gate9_root_type=type(audit).__name__, prompt_rows=8, prompt_length=2048,
                  prefill_budget=16384, prefix_caching=False,
                  open_checks=['installed vLLM accepts chunked-prefill=False with this budget',
                               'exact replay and scheduled-batch assertions on real requests',
                               'probe JIT, output checks and PTX/SASS checks',
                               'physical GPU overlap; Graph dispatch is not evidence',
                               'new fair performance baseline; historical baseline not reused as denominator'])
    (out / 'analysis').mkdir(exist_ok=True)
    (out / 'analysis/gate0.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
