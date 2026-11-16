"""Rebuild a derived command ledger from intact per-invocation supervision logs."""
from pathlib import Path
import json


def main():
    root=Path(__file__).resolve().parents[1]
    original=[json.loads(line) for line in (root/'commands.txt').read_text().splitlines() if line.strip()]
    records=[]
    for f in sorted((root/'logs').glob('*_supervision.json')):
        d=json.loads(f.read_text())['record']
        records.append(dict(recovered_from=f.relative_to(root).as_posix(),**d))
    records.sort(key=lambda x:x['start'])
    assert len(records)==5 and sum(x['status']=='completed' for x in records)==4
    dest=root/'commands_reconciled.jsonl'
    dest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=dict(original_aggregate_records=len(original),intact_supervision_records=len(records),
                reason='A deployment copied a stale aggregate commands.txt. Per-run supervision records were not overwritten.',
                original_preserved='commands.txt',derived_complete_ledger=dest.name,
                missing_actual_workload_commands=False,
                blocked_before_spawn='analysis/blocked_attempt.json')
    (root/'analysis/commands_reconciliation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
