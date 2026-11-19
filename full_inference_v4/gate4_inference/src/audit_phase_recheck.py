"""Verify matching E2 work and export P1 overhead against the same cohort's P0."""
import csv
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        w=csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def audit(root, raw):
    models, probes, cases, metrics, overhead = {}, set(), [], [], []
    with (root/'lifecycle/request_envelopes.csv').open(encoding='utf-8') as stream:
        baselines=defaultdict(list)
        for r in csv.DictReader(stream):
            baselines[r['precision'],r['condition']].append(float(r['model_wall_ms']))
    for folder in sorted(raw.glob('gate44_*')):
        stem=folder.name
        with gzip.open(folder/(stem+'.json.gz'), 'rt', encoding='utf-8') as stream:
            d=json.load(stream)
        guard=json.loads((folder/(stem+'_supervision.json')).read_bytes())
        precision=d['config']['precision']
        protocol=d['gate44']
        condition=protocol['condition']
        if guard['status']!='COMPLETED' or d['invocation_status']!='COMPLETED' or d['config']['batch']!=8:
            raise ValueError(f'invalid run {stem}')
        probes.add(protocol['cubin_sha256'])
        for run,aux in zip(d['runs'],protocol['trials'],strict=True):
            actual=tuple(sorted((r['prompt_hash'],r['output_hash']) for r in run['request_summary']))
            if len(actual)!=8 or len({i[0] for i in actual})!=8 or run['total_output_tokens']!=2056 or len(run['graph_replays'])!=256:
                raise ValueError('incorrect request shape')
            if aux['condition'] in ('parallel','serial'):
                if aux['completed_probes']!=33 or any(not x['correctness_passed'] for x in aux['completions']):
                    raise ValueError('probe work mismatch')
            elif aux['completed_probes']:
                raise ValueError('unexpected work in control')
            identity=(actual,d['config_file_sha256'],d['prompt_manifest_hash'],d['gpu_uuid'])
            if models.setdefault(precision,identity)!=identity:
                raise ValueError('request output, configuration, device or manifest changed')
        profile=d['profile']
        cases.append(dict(run_id=stem,timestamp=guard['start'],precision=precision,condition=condition,profile=profile,
                          completed_model_requests=8*len(d['runs']),full_inference_runs=len(d['runs']),
                          generated_tokens_each_request=257, actual_decode_steps=256, work_validated=True,
                          probe_binary_sha256=protocol['cubin_sha256'],worker_sha256=protocol['wrapper_sha256'],
                          config_sha256=d['config_file_sha256'],prompt_manifest_sha256=d['prompt_manifest_hash'],
                          output_set_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest()))
        if profile=='P1':
            doc=json.loads((root/'metrics'/(stem+'.json')).read_bytes())
            wall=d['runs'][0]['full_model_wall_ms']
            reference=mean(baselines[precision,condition])
            overhead.append(dict(run_id=stem,timestamp=guard['start'],precision=precision,phase='lifecycle',condition=condition,
                                 p1_wall_ms=wall,mean_p0_wall_ms=reference,p1_over_p0_minus_one=wall/reference-1,
                                 measurement_scope='INDEPENDENT_E2_PROFILING_CHECK',evidence_status='DESCRIPTIVE_SINGLE_P1'))
            for r in doc['rows']:
                metrics.append(dict(run_id=stem,precision=precision,condition=condition,**r))
    if len(probes)!=1 or len(models)!=2 or len({(r['precision'],r['condition'],r['profile']) for r in cases})!=10:
        raise ValueError('incomplete or mismatched measurement matrix')
    if len(cases)!=10 or len(metrics)!=96:
        raise ValueError('unexpected duplicate run or metric')
    write(root/'run_inventory.csv',cases)
    write(root/'device_metrics.csv',metrics)
    write(root/'p1_overhead.csv',overhead)


if __name__=='__main__':
    audit(Path(sys.argv[1]),Path(sys.argv[2]))
