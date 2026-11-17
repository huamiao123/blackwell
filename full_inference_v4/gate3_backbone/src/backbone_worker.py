"""One complete inference diagnostic with read-only graph/parameter metadata."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
sys.path.insert(0, str(V4 / 'gate3_attribution/src'))
import run_full_inference as original
import diagnostic_worker as previous
from graph_binding import GraphObserver, tensor_catalog
from common import file_hash

OBSERVER = None
CATALOG = []


class BackboneHooks(previous.DiagnosticHooks):
    def install(self):
        global CATALOG
        super().install()
        CATALOG = tensor_catalog(self.runner.model, self.torch)
        modules = {t['module'] for t in CATALOG}
        if len(modules) != 144:
            raise RuntimeError('expected four projections in 36 layers, got ' + str(len(modules)))
        OBSERVER.bind(CATALOG)
        base_replay = self.torch.cuda.CUDAGraph.replay

        def replay(graph, *args, **kwargs):
            if self.state is not None:
                OBSERVER.replay(graph, getattr(self.local, 'iteration', None))
            return base_replay(graph, *args, **kwargs)

        self.torch.cuda.CUDAGraph.replay = replay


def main():
    global OBSERVER
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--profile', choices=['P2'], default='P2')
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print('CPU plan only: read-only post-instantiate graph metadata')
        return
    original.require_permit(a.config, a.run_id)
    import torch
    OBSERVER = GraphObserver(torch)
    save = original.save

    def backbone_save(path, data, **kwargs):
        if isinstance(data, dict) and data.get('schema') in ['v4.1', 'v4.1_failure']:
            data['backbone_metadata'] = dict(worker_sha256=file_hash(HERE),
                binding_source_sha256=file_hash(HERE.with_name('graph_binding.py')),
                tensor_catalog=CATALOG, captures=OBSERVER.snapshots, replays=OBSERVER.replays,
                scope='read-only host metadata; no copied GPU weight values; no graph modification')
        save(path, data, **kwargs)

    original.save = backbone_save
    previous.DiagnosticHooks = BackboneHooks
    try:
        previous.main()
    finally:
        OBSERVER.close()


if __name__ == '__main__':
    main()
