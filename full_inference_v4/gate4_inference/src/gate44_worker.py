"""Inject an independently validated probe at E2 GPU dispatch boundaries."""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve()
V4 = HERE.parents[2]
sys.path[:0] = [str(V4 / 'src'), str(V4 / 'gate4_spatial/src')]
import run_full_inference as original
from common import file_hash
from g4_common import SPATIAL, load
from probe_reference import validate_outputs, work_accounting
from qkv_overlap_pilot import initialize_probe_buffers, short_probe_spec
from replay import checked

GRID = 64
DECODE_ITERATIONS = tuple(range(33, 65))
PAIRED_SEQUENCES = (
    ('a_alone', 'empty', 'serial', 'parallel'),
    ('serial', 'parallel', 'empty', 'a_alone'),
    ('parallel', 'serial', 'a_alone', 'empty'),
    ('empty', 'a_alone', 'parallel', 'serial'),
    ('serial', 'a_alone', 'empty', 'parallel'),
)


def targets(phase):
    if phase == 'prefill':
        return (0,)
    if phase == 'decode':
        return DECODE_ITERATIONS
    return (0, *DECODE_ITERATIONS)


class Gate44Hooks(original.Hooks):
    def install(self):
        from cuda.bindings import driver

        super().install()
        self.driver = driver
        self.pilot_dispatch = self.core.model_executor.execute_model
        self.pilot_replay = self.torch.cuda.CUDAGraph.replay
        self.core.model_executor.execute_model = self.dispatch
        self.torch.cuda.CUDAGraph.replay = self.replay_entry
        cubin = SPATIAL / 'build/offline_sm120_v2/probes_sm120.cubin'
        self.cubin_sha = file_hash(cubin)
        manifest = load(SPATIAL / 'analysis/offline_compile/compile_manifest.json')
        if manifest['cubin_sha256'] != self.cubin_sha:
            raise ValueError('unvalidated probe binary')
        self.spec = short_probe_spec('tensor_light')
        self.sources, output, self.tiles, self.rounds = initialize_probe_buffers(self.torch, self.spec)
        self.elements = output.numel()
        self.module = checked(driver.cuModuleLoad(os.fsencode(cubin)))
        self.function = checked(driver.cuModuleGetFunction(self.module, b'tensor_light'))
        self.stream = self.torch.cuda.Stream()
        self.ctx = int(checked(driver.cuCtxGetCurrent()))
        self.buffers = self.torch.full((len(self.conditions), len(self.targets), self.elements), float('nan'),
                                       dtype=self.torch.float32, device='cuda')
        self.trial = -1
        self.previous_state = None
        self.measurements = []
        self.pending = {}
        self.events = [[self.event_set() for _ in self.targets] for _ in self.conditions]
        self.torch.cuda.synchronize()
        self.baseline = self.validate_alone(output)
        self.__class__.active = self

    def event_set(self):
        events = tuple(self.torch.cuda.Event(enable_timing=i in (1, 2)) for i in range(3))
        for event in events:
            event.record()
        return events

    def launch(self, output):
        driver = self.driver
        pointers = tuple(t.data_ptr() for t in (*self.sources, output))
        arguments = (pointers + (self.tiles, self.rounds),
                     (ctypes.c_uint64,) * len(pointers) + (ctypes.c_uint32, ctypes.c_uint32))
        checked(driver.cuLaunchKernel(
            self.function, GRID, 1, 1, self.spec['threads'], 1, 1, self.spec['shared_bytes'],
            driver.CUstream(init_value=self.stream.cuda_stream), arguments, 0))

    def validate_alone(self, output):
        samples = []
        for repeat in range(6):
            output.fill_(float('nan'))
            ready, begin, end = self.event_set()
            self.stream.wait_event(ready)
            begin.record(self.stream)
            self.launch(output)
            end.record(self.stream)
            end.synchronize()
            evidence = validate_outputs(self.spec, output.cpu().numpy())
            if repeat:
                samples.append(begin.elapsed_time(end) * 1000)
        return dict(gpu_us=samples, median_us=statistics.median(samples), **evidence)

    def active_trial(self):
        if self.state is not self.previous_state:
            self.previous_state = self.state
            self.trial += 1
            if self.pending or self.trial >= len(self.conditions):
                raise RuntimeError('unexpected trial boundary')
        return self.trial

    def arm(self, iteration, phase, ready_recorded=False):
        trial = self.active_trial()
        condition = self.conditions[trial]
        if condition == 'a_alone' or iteration not in self.targets:
            return
        if iteration in self.pending or int(checked(self.driver.cuCtxGetCurrent())) != self.ctx:
            raise RuntimeError('duplicate injection or changed CUDA context')
        slot = self.targets.index(iteration)
        ready, begin, end = self.events[trial][slot]
        main = self.torch.cuda.current_stream()
        if main.cuda_stream == self.stream.cuda_stream:
            raise RuntimeError('probe and model use the same stream')
        if not ready_recorded:
            ready.record(main)
        self.stream.wait_event(ready)
        begin.record(self.stream)
        with self.nvtx(f'G44_B|{phase}|{iteration}|{slot}'):
            if condition != 'empty':
                self.launch(self.buffers[trial, slot])
        end.record(self.stream)
        if condition == 'serial':
            main.wait_event(end)
        self.pending[iteration] = dict(phase=phase, slot=slot, model_stream_handle=main.cuda_stream,
                                       aux_stream_handle=self.stream.cuda_stream)

    def dispatch(self, *args, **kwargs):
        if self.state is not None:
            trial = self.active_trial()
            schedule = args[0] if args else kwargs.get('scheduler_output')
            row = self.schedule_ids[id(schedule)]
            iteration = row['scheduler_iteration']
            if row['stage'] == 'PREFILL' and iteration in self.targets:
                if row['active_request_count'] != 8:
                    raise RuntimeError('unexpected E2 Prefill batch')
                if self.conditions[trial] in ('parallel', 'empty'):
                    ready = self.events[trial][self.targets.index(iteration)][0]
                    # Capture prerequisites before forwarding, then let the CPU
                    # enqueue the real Prefill work before it submits the probe.
                    ready.record(self.torch.cuda.current_stream())
                    result = self.pilot_dispatch(*args, **kwargs)
                    self.arm(iteration, 'PREFILL', ready_recorded=True)
                    return result
                self.arm(iteration, 'PREFILL')
        return self.pilot_dispatch(*args, **kwargs)

    @staticmethod
    def replay_entry(graph, *args, **kwargs):
        hooks = Gate44Hooks.active
        if hooks.state is not None:
            iteration = getattr(hooks.local, 'iteration', None)
            if isinstance(iteration, int) and hooks.state.iterations[iteration]['stage'] == 'DECODE':
                if iteration in hooks.targets and hooks.state.iterations[iteration]['active_request_count'] != 8:
                    raise RuntimeError('unexpected E2 Decode batch')
                hooks.arm(iteration, 'DECODE')
        return hooks.pilot_replay(graph, *args, **kwargs)

    def verify_trial(self):
        condition = self.conditions[self.trial]
        scheduled = set() if condition == 'a_alone' else set(self.targets)
        if set(self.pending) != scheduled:
            raise RuntimeError('wrong number of E2 injections')
        completed = []
        for iteration, row in self.pending.items():
            _, begin, end = self.events[self.trial][row['slot']]
            elapsed_us = begin.elapsed_time(end) * 1000
            if condition != 'empty':
                actual = self.buffers[self.trial, row['slot']].cpu().numpy()
                evidence = validate_outputs(self.spec, actual)
                row.update(evidence, output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
            completed.append(dict(iteration=iteration, elapsed_us=elapsed_us, **row))
        self.measurements.append(dict(trial=self.trial, condition=condition, completions=completed,
                                      completed_probes=len(completed) if condition != 'empty' else 0))
        self.pending.clear()

    def restore(self):
        super().restore()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', choices=('P0', 'P1', 'P2'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('prefill', 'decode', 'lifecycle'), required=True)
    parser.add_argument('--condition', choices=('parallel', 'serial', 'empty', 'a_alone', 'paired'), required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    original.require_permit(args.config, args.run_id)
    Gate44Hooks.conditions = (tuple(condition for block in PAIRED_SEQUENCES for condition in block)
                              if args.condition == 'paired' else (args.condition,))
    Gate44Hooks.targets = targets(args.phase)
    original.Hooks = Gate44Hooks
    prepare, save = original.prepare, original.save

    def configure(path):
        c, prompts = prepare(path)
        if (c['experiment'], c['batch'], c['decode_steps']) != ('E2', 8, 256):
            raise ValueError('expected frozen E2 workload')
        c['trials'] = len(Gate44Hooks.conditions)
        return c, prompts

    def save_measurements(path, data, **kwargs):
        if data.get('schema') == 'v4.1':
            hooks = Gate44Hooks.active
            if len(data['runs']) > len(hooks.measurements):
                hooks.verify_trial()
            data['gate44'] = dict(phase=args.phase, condition=args.condition, targets=hooks.targets,
                                  condition_order=hooks.conditions, standalone=hooks.baseline,
                                  prefill_parallel_entry='AFTER_HOST_PREFILL_DISPATCH_WITH_PRE_DISPATCH_EVENT',
                                  probe=work_accounting(hooks.spec, GRID), cubin_sha256=hooks.cubin_sha,
                                  wrapper_sha256=file_hash(HERE), trials=hooks.measurements)
        save(path, data, **kwargs)

    original.prepare, original.save = configure, save_measurements
    original.execute(args.config, args.run_id, args.profile, args.output)


if __name__ == '__main__':
    main()
