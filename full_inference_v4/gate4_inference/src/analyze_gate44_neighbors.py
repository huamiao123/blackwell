"""Locate actual primary kernels intersecting a validated E2 probe (P2 SQLite)."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys


def union_ns(intervals):
    total = 0
    last = -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, last))
        last = max(last, end)
    return total


def neighbors(db, verified):
    if verified['condition'] not in ('serial', 'parallel'):
        raise ValueError('measured controls required')
    names = {}
    rows = []
    for v in verified['rows']:
        start, end = v['probe_start_ns'], v['probe_end_ns']
        probe, = db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=? AND start=?',
                            (v['probe_correlation'], start))
        api, = db.execute('SELECT r.start,r.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                          'JOIN StringIds s ON s.id=r.nameId WHERE r.correlationId=?',
                          (v['probe_correlation'],))
        if (probe['end'] != end or probe['streamId'] != v['probe_stream_id']
                or probe['contextId'] != v['context_id'] or start < api['start']):
            raise ValueError('ambiguous probe identity or time')
        hits = []
        for k in db.execute('SELECT k.*,s.value AS short,d.value AS full FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                            'JOIN StringIds s ON s.id=k.shortName JOIN StringIds d ON d.id=k.demangledName '
                            'WHERE k.start<? AND k.end>? AND k.globalPid=? AND k.contextId=? AND k.deviceId=?',
                            (end, start, probe['globalPid'], probe['contextId'], probe['deviceId'])):
            if (k['streamId'] not in v['primary_stream_ids'] or k['start'] < v['primary_first_ns']
                    or k['end'] > v['primary_last_ns']
                    or (v['phase'] == 'DECODE' and k['graphId'] != v['primary_graph_id'])):
                continue
            names[k['demangledName']] = dict(short=k['short'], demangled=k['full'])
            hits.append(dict(name_id=k['demangledName'], start_ns=k['start'], end_ns=k['end'],
                             overlap_ns=min(end,k['end'])-max(start,k['start']), grid=[k['gridX'],k['gridY'],k['gridZ']],
                             block=[k['blockX'],k['blockY'],k['blockZ']], registers=k['registersPerThread'],
                             static_smem=k['staticSharedMemory'], dynamic_smem=k['dynamicSharedMemory'],
                             graph=k['graphId'], node=k['graphNodeId'], stream=k['streamId']))
        if union_ns((max(start,h['start_ns']),min(end,h['end_ns'])) for h in hits) != v['overlap_ns']:
            raise ValueError('candidate neighbors do not reproduce independently verified target intersection')
        rows.append(dict(iteration=v['iteration'], phase=v['phase'], probe_start_ns=start, probe_end_ns=end,
                         primary_first_ns=v['primary_first_ns'], primary_last_ns=v['primary_last_ns'],
                         overlap_ns=v['overlap_ns'], api_name=api['value'], api_start_ns=api['start'],
                         api_end_ns=api['end'], hits=sorted(hits,key=lambda h:h['start_ns'])))
    return dict(run_id=verified['run_id'], precision=verified['precision'], condition=verified['condition'],
                uuid=verified['device_uuid'], names=names, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, help='full_inference_v4 tree containing src, sqlite, raw and logs')
    parser.add_argument('run')
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'gate4_inference/src'))
    from analyze_gate44_trace import analyze
    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))
    with sqlite3.connect((args.root/'sqlite'/(args.run+'.sqlite')).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        checked = analyze(db,load(args.root/'raw'/(args.run+'.json')),load(args.root/'logs'/(args.run+'_supervision.json')))
        print(json.dumps(neighbors(db,checked)))
