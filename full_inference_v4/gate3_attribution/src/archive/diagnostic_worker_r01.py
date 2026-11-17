"""One full-inference P2 diagnostic; frozen Gate-2 worker is imported, not edited.

Adds executor sampling and actual LM-head apply ranges outside model graphs.
Imports remain CPU-only until the original permit check in execute().
"""
import argparse
import hashlib
import time
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path.insert(0, str(V4 / 'src'))
import run_full_inference as original
from common import file_hash


def sample_row(pending, request_ids):
    if len(pending) != 1:
        raise RuntimeError('ambiguous execute/sample association')
    row = pending[0]
    expected = {r['request_id'] for r in row['requests'] if r['scheduled_tokens']}
    if set(request_ids) != expected:
        raise RuntimeError('sample input batch differs from scheduled requests')
    return row


ACTIVE_HOOKS = None


class DiagnosticHooks(original.Hooks):
    def install(self):
        global ACTIVE_HOOKS
        super().install()
        self.pending = []
        self.sampling_row = None
        self.gate3_events = []
        executor = self.core.model_executor
        runner = executor.driver_worker.worker.model_runner
        module = type(runner).__module__
        if module != 'vllm.v1.worker.gpu.model_runner':
            raise RuntimeError('unexpected runner: ' + module)
        self.runner = runner
        model = runner.model
        head = model.lm_head
        self.gate3_metadata = dict(runner_class=module + '.' + type(runner).__name__,
            runner_source_file=sys.modules[module].__file__,
            runner_source_sha256=file_hash(sys.modules[module].__file__),
            head_quant_method=type(head.quant_method).__module__ + '.' + type(head.quant_method).__name__,
            head_weight_shape=list(head.weight.shape), head_weight_dtype=str(head.weight.dtype),
            note='Only shape/type metadata; no tensor data copied to CPU. No layer forward hooks inserted.')
        dispatch = executor.execute_model
        sample = executor.sample_tokens
        logits = model.compute_logits
        head_apply = head.quant_method.apply

        def dispatch_hook(*args, **kwargs):
            if self.state is not None:
                s = args[0] if args else kwargs['scheduler_output']
                row = self.schedule_ids.get(id(s))
                if row is None or not row['active_request_count']:
                    raise RuntimeError('unmapped/nonactive model dispatch')
                if self.pending:
                    raise RuntimeError('more than one unsampled dispatch')
                self.pending.append(row)
            return dispatch(*args, **kwargs)

        def sample_hook(*args, **kwargs):
            if self.state is None:
                return sample(*args, **kwargs)
            runner_state = runner.execute_model_state
            if runner_state is None:
                raise RuntimeError('missing runner sampling state')
            row = sample_row(self.pending, runner_state.input_batch.req_ids)
            iteration, phase = row['scheduler_iteration'], row['stage']
            self.sampling_row = row
            event = dict(iteration=iteration, phase=phase,
                         req_ids=list(runner_state.input_batch.req_ids), cpu_start_ns=time.perf_counter_ns())
            self.gate3_events.append(event)
            try:
                with self.nvtx(f'SAMPLE_DISPATCH|{iteration}|{phase}'):
                    return sample(*args, **kwargs)
            finally:
                event['cpu_end_ns'] = time.perf_counter_ns()
                self.sampling_row = None
                self.pending.pop(0)

        def logits_hook(hidden_states, *args, **kwargs):
            row = self.sampling_row
            if self.state is None:
                return logits(hidden_states, *args, **kwargs)
            if row is None:
                raise RuntimeError('logits outside verified sample association')
            self.gate3_events[-1]['logits_input_shape'] = list(hidden_states.shape)
            self.gate3_events[-1]['logits_input_dtype'] = str(hidden_states.dtype)
            with self.nvtx(f'LOGITS|{row["scheduler_iteration"]}|{row["stage"]}'):
                return logits(hidden_states, *args, **kwargs)

        def head_hook(layer, x, *args, **kwargs):
            row = self.sampling_row
            if self.state is None:
                return head_apply(layer, x, *args, **kwargs)
            if row is None or layer is not head:
                raise RuntimeError('unmapped head apply')
            event = self.gate3_events[-1]
            event['head_apply_calls'] = event.get('head_apply_calls', 0) + 1
            with self.nvtx(f'LOGICAL_OP|LM_HEAD|{row["scheduler_iteration"]}|{row["stage"]}'):
                return head_apply(layer, x, *args, **kwargs)

        self.extra_originals = (executor, dispatch, sample, model, logits, head.quant_method, head_apply)
        executor.execute_model = dispatch_hook
        executor.sample_tokens = sample_hook
        model.compute_logits = logits_hook
        head.quant_method.apply = head_hook
        ACTIVE_HOOKS = self

    def restore(self):
        if hasattr(self, 'extra_originals'):
            executor, dispatch, sample, model, logits, method, head_apply = self.extra_originals
            executor.execute_model = dispatch
            executor.sample_tokens = sample
            model.compute_logits = logits
            method.apply = head_apply
        super().restore()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--profile', choices=['P2'], default='P2')
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print('CPU plan only: E2 NVFP4 complete workload + sampling/head ranges')
        return
    original.Hooks = DiagnosticHooks
    save = original.save

    def diagnostic_save(path, d, **kwargs):
        if isinstance(d, dict) and d.get('schema') == 'v4.1':
            d['gate3_diagnostic'] = dict(wrapper_sha256=file_hash(HERE),
                runtime_metadata=ACTIVE_HOOKS.gate3_metadata if ACTIVE_HOOKS else None,
                sample_events=ACTIVE_HOOKS.gate3_events if ACTIVE_HOOKS else [],
                measurement_scope='P2 diagnostic only; not formal performance or same-SM evidence')
        save(path, d, **kwargs)

    original.save = diagnostic_save
    original.execute(a.config, a.run_id, a.profile, a.output)


if __name__ == '__main__':
    main()
