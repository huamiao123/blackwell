"""Real-model Graph Decode, finite private-stream work, raw paired observations.

Never run directly on a shared GPU: use run_supervised.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def prompt_ids(batch, length=2048):
    return [[1000 + ((104729 * r + 37 * i + (i * i) % 8192) % (151936 - 1000))
             for i in range(length)] for r in range(batch)]


def sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', choices=['bf16', 'nvfp4'], required=True)
    ap.add_argument('--batch', type=int, choices=[1, 8], default=8)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--probe', choices=['none', 'fp32', 'dram', 'tensor'], default='none')
    ap.add_argument('--chunks', type=int, default=128)
    ap.add_argument('--reps', type=int, default=800000)
    ap.add_argument('--inner', type=int, default=512)
    ap.add_argument('--ctas', type=int, default=8)
    ap.add_argument('--modes', default='alone_a')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from probes import Probe
    root = Path.home() / 'fp4-llm'
    dest = Path(a.output).resolve()
    assert dest.is_relative_to(root) and not dest.exists()
    modes = a.modes.split(',')
    assert set(modes) <= {'alone_a', 'alone_b', 'serial_ab', 'serial_ba', 'parallel_ab', 'parallel_ba', 'empty_ab', 'empty_ba'}
    assert a.probe != 'none' or modes == ['alone_a']
    assert not a.profile or (a.trials == 1 and len(modes) == 1)
    ids = prompt_ids(a.batch)
    assert len({tuple(row) for row in ids}) == a.batch
    prompts = [{'prompt_token_ids': row} for row in ids]
    settings = dict(max_model_len=4096, gpu_memory_utilization=.85,
                    linear_backend='cutlass', enforce_eager=False,
                    kv_cache_dtype='fp8_e4m3', attention_backend='FLASHINFER',
                    disable_log_stats=True, enable_prefix_caching=False,
                    enable_chunked_prefill=False, max_num_batched_tokens=16384,
                    max_num_seqs=8, cudagraph_capture_sizes=[1, 8], seed=0)
    model = root / 'models' / ('Qwen3-8B-BF16' if a.precision == 'bf16' else 'Qwen3-8B-NVFP4')
    llm = LLM(model=str(model), **settings)
    eng = llm.llm_engine
    assert type(eng.engine_core).__name__ == 'InprocClient'
    core = eng.engine_core.engine_core
    result = dict(schema=1, args=vars(a), settings=settings, prompt_ids=ids,
                  prompt_sha256=sha(ids), prompt_row_sha256=[sha(x) for x in ids],
                  pid=os.getpid(), model=str(model), torch_version=torch.__version__,
                  async_scheduling=core.async_scheduling,
                  timing='first actual Decode Graph replay -> output completion plus A-stream event',
                  runs=[], source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                        for f in Path(__file__).parent.glob('*.py')})
    props = torch.cuda.get_device_properties(0)
    result['gpu'] = {k: getattr(props, k, None) for k in
                     ['name', 'major', 'minor', 'multi_processor_count', 'total_memory', 'L2_cache_size']}
    probe = None if a.probe == 'none' else Probe(a.probe, a.chunks, a.reps, a.inner, a.ctas)
    empty = Probe('empty', a.chunks) if probe else None
    if probe:
        result['probe'] = probe.describe()
        result['probe']['compiler'] = probe.evidence(dest.parent.parent / 'analysis' / (dest.stem + '_compiler'))
    state = dict(measured=False)
    original_replay = torch.cuda.CUDAGraph.replay
    original_schedule = core.scheduler.schedule
    original_process = eng.output_processor.process_outputs
    original_step = eng.step

    def launch_b():
        b = state['b']
        with torch.cuda.stream(b.stream):
            b.stream.wait_event(state['ready'])
            state['b0'].record(b.stream)
            t0 = time.perf_counter_ns()
            original_replay(b.graph)
            state['b_submit_ns'] = [t0, time.perf_counter_ns()]
            state['b1'].record(b.stream)

    def begin_first_replay():
        stream = torch.cuda.current_stream()
        state['a_stream'] = stream
        state['ready'].record(stream)
        state['joint_start_ns'] = time.perf_counter_ns()
        torch.cuda.nvtx.range_push('OVERLAP_JOINT|' + state['label'])
        if state['mode'] == 'serial_ba':
            launch_b()
            state['b1'].synchronize()
        state['a0'].record(stream)
        state['a_start_ns'] = time.perf_counter_ns()
        torch.cuda.nvtx.range_push('OVERLAP_A|' + state['label'])
        if state['mode'] in ['parallel_ba', 'empty_ba']:
            launch_b()
        state['first'] = False

    def replay_hook(graph, *args, **kwargs):
        if not (state['measured'] and state.get('decode', False)):
            return original_replay(graph, *args, **kwargs)
        if state['first']:
            begin_first_replay()
        state['replays'] += 1
        t0 = time.perf_counter_ns()
        ret = original_replay(graph, *args, **kwargs)
        if state['replays'] == 1:
            state['a_first_submit_ns'] = [t0, time.perf_counter_ns()]
            if state['mode'] in ['parallel_ab', 'empty_ab']:
                launch_b()
        return ret

    def schedule_hook(*args, **kwargs):
        before = {rid: (r.num_computed_tokens, r.num_prompt_tokens)
                  for rid, r in core.scheduler.requests.items()} if state['measured'] else {}
        s = original_schedule(*args, **kwargs)
        if state['measured']:
            decode = bool(s.num_scheduled_tokens) and all(
                before[r][0] >= before[r][1] and n == 1 for r, n in s.num_scheduled_tokens.items())
            state['decode'] = decode
            row = dict(batch=len(s.num_scheduled_tokens), scheduled=list(s.num_scheduled_tokens.values()),
                       computed_before=[before[r][0] for r in s.num_scheduled_tokens], decode=decode,
                       preempted=len(getattr(s, 'preempted_req_ids', [])))
            state['schedules'].append(row)
            if decode and not state.get('prefill_drained'):
                assert len(s.num_scheduled_tokens) == a.batch, 'staggered prefill/Decode'
                torch.cuda.synchronize()  # Only before measured A/B dispatch.
                state['prefill_drained'] = True
                state['request_ids'] = list(s.num_scheduled_tokens)
                if a.profile:
                    torch.cuda.profiler.start()
            if decode:
                torch.cuda.nvtx.mark('A_SCHEDULE|' + state['label'])
        return s

    def process_hook(outputs, *args, **kwargs):
        ret = original_process(outputs, *args, **kwargs)
        if state['measured']:
            for o in outputs:
                state['completed'][o.request_id] = state['completed'].get(o.request_id, 0) + len(o.new_token_ids)
            if 'a_start_ns' in state:
                done = min(state['completed'].get(r, 0) for r in state['request_ids']) - 1
                while len(state['step_rows']) < done:
                    t = time.perf_counter_ns()
                    prev = state['step_rows'][-1]['end_ns'] if state['step_rows'] else state['a_start_ns']
                    state['step_rows'].append(dict(step=len(state['step_rows']) + 1, end_ns=t, latency_ms=(t - prev) / 1e6))
                    torch.cuda.nvtx.mark('A_OUTPUT|' + state['label'] + '|' + str(len(state['step_rows'])))
        return ret

    def step_hook(*args, **kwargs):
        ret = original_step(*args, **kwargs)
        if state['measured'] and len(state['step_rows']) == a.steps and 'a_end_ns' not in state:
            state['a1'].record(state['a_stream'])
            state['a1'].synchronize()  # Must NOT join unrelated B here.
            state['a_end_ns'] = time.perf_counter_ns()
            state['b_done_at_a_end'] = state['b1'].query() if state['b'] and 'b_submit_ns' in state else None
            torch.cuda.nvtx.range_pop()
            if state['mode'] == 'serial_ab':
                launch_b()
        return ret

    core.scheduler.schedule = schedule_hook
    eng.output_processor.process_outputs = process_hook
    eng.step = step_hook
    torch.cuda.CUDAGraph.replay = replay_hook
    print('OVERLAP_MODEL_READY', a.precision, a.batch, 'prefix_cache=False', flush=True)
    warm = SamplingParams(max_tokens=33, temperature=0., ignore_eos=True, seed=0)
    for _ in range(2):
        llm.generate(prompts, warm, use_tqdm=False)
    torch.cuda.synchronize()
    try:
        for repeat in range(a.trials):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            if repeat % 2:
                order = list(reversed(order))
            for mode in order:
                b = empty if mode.startswith('empty') else (probe if mode != 'alone_a' else None)
                if b:
                    b.reset()
                state.clear()
                state.update(measured=mode != 'alone_b', mode=mode, b=b, first=True,
                             label=f'{dest.stem}|r{repeat}|{mode}', decode=False,
                             replays=0, schedules=[], step_rows=[], completed={})
                for name in ['ready', 'a0', 'a1', 'b0', 'b1']:
                    state[name] = torch.cuda.Event(enable_timing=True)
                call_start = time.perf_counter_ns()
                outputs = llm.generate(prompts, SamplingParams(max_tokens=1 if mode == 'alone_b' else a.steps + 1,
                                       temperature=0., ignore_eos=True, seed=0), use_tqdm=False)
                if mode == 'alone_b':
                    torch.cuda.synchronize()
                    state['ready'].record(torch.cuda.current_stream())
                    state['joint_start_ns'] = time.perf_counter_ns()
                    launch_b()
                else:
                    assert 'a_end_ns' in state, 'A window not completed'
                if b:
                    state['b1'].synchronize()
                joint_end = time.perf_counter_ns()
                if mode != 'alone_b':
                    torch.cuda.nvtx.range_pop()
                if a.profile:
                    torch.cuda.profiler.stop()
                state['measured'] = False
                torch.cuda.synchronize()  # Outside measured windows, after both completions.
                counts = [len(o.outputs[0].token_ids) for o in outputs]
                expected = 1 if mode == 'alone_b' else a.steps + 1
                assert counts == [expected] * a.batch, counts
                decode_rows = [s for s in state['schedules'] if s['decode']]
                if mode != 'alone_b':
                    assert len(decode_rows) == a.steps and len(state['step_rows']) == a.steps
                    assert state['replays'] == a.steps, ('Graph fallback/replay mismatch', state['replays'])
                    assert all(s['batch'] == a.batch and s['scheduled'] == [1] * a.batch for s in decode_rows)
                    assert not any(s['preempted'] for s in state['schedules']), 'preemption'
                    nonempty = [s for s in state['schedules'] if s['batch']]
                    print('OVERLAP_SCHEDULER', len(state['schedules']), 'calls', len(nonempty),
                          'nonempty', len(decode_rows), 'decode', flush=True)
                    assert len(nonempty) == a.steps + 1, ('unexpected nonempty scheduler shape', nonempty[:2], nonempty[-2:])
                row = dict(repeat=repeat, mode=mode, order=order, label=state['label'], output_counts=counts,
                           output_sha256=sha([o.outputs[0].token_ids for o in outputs]),
                           replays=state['replays'], scheduler=state['schedules'], steps=state['step_rows'],
                           joint_wall_ms=(joint_end - state['joint_start_ns']) / 1e6,
                           generate_call_wall_ms=(joint_end - call_start) / 1e6,
                           b_done_at_a_end=state.get('b_done_at_a_end'),
                           a_first_submit_ns=state.get('a_first_submit_ns'), b_submit_ns=state.get('b_submit_ns'))
                row.update(a_start_ns=state.get('a_start_ns'), a_end_ns=state.get('a_end_ns'),
                           joint_start_ns=state['joint_start_ns'], joint_end_ns=joint_end)
                if mode != 'alone_b':
                    wall = (state['a_end_ns'] - state['a_start_ns']) / 1e6
                    lat = sorted(x['latency_ms'] for x in state['step_rows'])
                    row.update(a_wall_ms=wall, mean_step_ms=wall / a.steps,
                               p50_step_ms=statistics.median(lat), p95_step_ms=lat[int(.95 * (len(lat) - 1))],
                               tokens_per_s=a.batch * a.steps * 1000 / wall,
                               a_event_ms=state['a0'].elapsed_time(state['a1']),
                               actual_batches=sorted({s['batch'] for s in decode_rows}))
                if b:
                    row.update(b_span_ms=state['b0'].elapsed_time(state['b1']),
                               b_completion_from_ready_ms=state['ready'].elapsed_time(state['b1']),
                               probe=b.describe(), validation=b.validate(root / 'cache/resource_overlap_v1'))
                result['runs'].append(row)
                dest.write_text(json.dumps(result, indent=2))
                print('OVERLAP_TRIAL', repeat, mode, 'A_ms', row.get('a_wall_ms'),
                      'B_ms', row.get('b_span_ms'), 'joint_ms', row['joint_wall_ms'], flush=True)
    except BaseException as error:
        failed = dict(error=repr(error), args=vars(a), settings=settings,
                      schedules=state.get('schedules'), steps=state.get('step_rows'),
                      replays=state.get('replays'), completed=state.get('completed'))
        dest.with_name(dest.stem + '_failed.json').write_text(json.dumps(failed, indent=2))
        raise
    finally:
        torch.cuda.CUDAGraph.replay = original_replay
        core.scheduler.schedule = original_schedule
        eng.output_processor.process_outputs = original_process
        eng.step = original_step
        eng.engine_core.shutdown()
    print('OVERLAP_SAVED', dest, flush=True)


if __name__ == '__main__':
    main()
