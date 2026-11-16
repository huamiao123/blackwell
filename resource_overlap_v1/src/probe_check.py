"""Gate 2: bounded standalone work, correctness and instruction evidence."""
import argparse
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--kind', choices=['fp32', 'dram', 'tensor', 'empty'], required=True)
    ap.add_argument('--chunks', type=int, default=32)
    ap.add_argument('--reps', type=int, default=200000)
    ap.add_argument('--inner', type=int, default=128)
    ap.add_argument('--ctas', type=int, default=8)
    ap.add_argument('--trials', type=int, default=5)
    a = ap.parse_args()
    import torch
    from probes import Probe
    dest = Path(a.output).resolve()
    assert dest.is_relative_to(Path.home() / 'fp4-llm') and not dest.exists()
    p = Probe(a.kind, a.chunks, a.reps, a.inner, a.ctas)
    result = dict(args=vars(a), description=p.describe(),
                  compiler=p.evidence(dest.parent.parent / 'analysis' / (dest.stem + '_compiler')), runs=[])
    for rep in range(a.trials):
        p.reset()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        t0 = time.perf_counter_ns()
        with torch.cuda.stream(p.stream):
            start.record(p.stream)
            p.graph.replay()
            end.record(p.stream)
        end.synchronize()
        t1 = time.perf_counter_ns()
        row = dict(repeat=rep, event_ms=start.elapsed_time(end), wall_ms=(t1 - t0) / 1e6,
                   validation=p.validate(Path.home() / 'fp4-llm/cache/resource_overlap_v1'))
        result['runs'].append(row)
        dest.write_text(json.dumps(result, indent=2))
        print('PROBE_CHECK', a.kind, rep, row['event_ms'], 'ms', flush=True)


if __name__ == '__main__':
    main()
