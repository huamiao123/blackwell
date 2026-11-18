"""One isolated QKV/FP32 block, with tracing only for the P2 diagnostic."""

import ctypes
import os
from pathlib import Path
import random

from cuda_stream_backend import CudaStreamBackend, Executable
from g4_common import SPATIAL, sha
from plan import MODES, probe_specs
from probe_reference import validate_outputs, work_accounting
from replay import checked
from stream_protocol import submit
from validate_probe_gpu import require_permit


def short_probe_spec(symbol):
    probe_kind, variant = symbol.split('_', 1)
    kind = {'fp32': 'fp32_fma', 'tensor': 'bf16_mma'}[probe_kind]
    spec = next(row.copy() for row in probe_specs() if row['probe'] == kind and row['variant'] == variant)
    if kind == 'fp32_fma':
        spec.update(items=65536, rounds=8, expected_flops=16_777_216)
    else:
        spec.update(tiles=256, mma_rounds=128, expected_flops=256 * 128 * 2 * 16 ** 3)
    return spec


def initialize_probe_buffers(torch, spec):
    if spec['probe'] == 'fp32_fma':
        elements = spec['items'] * 16
        source = (0.25 + (torch.arange(elements, dtype=torch.int32) % 64).float() / 128).to('cuda')
        return [source], torch.empty_like(source), spec['items'], spec['rounds']
    index = (torch.arange(16)[:, None] + torch.arange(spec['k_staging'])[None, :]) % 4
    inputs = [(base + stride * index).to('cuda', torch.bfloat16).contiguous()
              for base, stride in ((0.5, 0.125), (0.25, 0.0625))]
    output = torch.empty(spec['tiles'] * 256, dtype=torch.float32, device='cuda')
    return inputs, output, spec['tiles'], spec['mma_rounds']


def pilot(torch, driver, run_dir, worker_path, case, graph, output, reference, cubin_sha256,
          profile='P2', block=0, probe_grid=128, compare_grids=False, probe_symbol='fp32_light'):
    if profile == 'P2':
        from nvtx import annotate
    scope = 'standalone_qkv_p0' if profile == 'P0' else 'standalone_qkv_overlap'

    uri = SPATIAL / 'build/offline_sm120_v2/probes_sm120.cubin'
    if sha(uri) != cubin_sha256:
        raise RuntimeError('calibrated CUBIN hash mismatch')
    spec = short_probe_spec(probe_symbol)
    grids = [probe_grid]
    if compare_grids:
        if profile != 'P0':
            raise ValueError('paired grid comparison is an unprofiled experiment')
        low_grid = probe_grid if probe_grid < 128 else 32
        grids = [low_grid, 128] if block % 2 == 0 else [128, low_grid]
    sources, destination, problem_size, rounds = initialize_probe_buffers(torch, spec)
    module = checked(driver.cuModuleLoad(os.fsencode(uri)))
    function = checked(driver.cuModuleGetFunction(module, probe_symbol.encode()))
    context_id = int(checked(driver.cuCtxGetCurrent()))
    attributes = driver.CUfunction_attribute
    registers = checked(driver.cuFuncGetAttribute(attributes.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    static_shared = checked(driver.cuFuncGetAttribute(attributes.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function))
    capacity = checked(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, spec['threads'], spec['shared_bytes']))
    addresses = tuple(tensor.data_ptr() for tensor in [*sources, destination])
    args = (addresses + (problem_size, rounds), (ctypes.c_uint64,) * len(addresses) + (ctypes.c_uint32, ctypes.c_uint32))
    capture = torch.cuda.Stream()
    capture_driver = driver.CUstream(init_value=capture.cuda_stream)
    a = Executable(driver.CUgraphExec(init_value=graph.raw_cuda_graph_exec()), context_id, graph)
    probes = {}
    torch.cuda.synchronize()
    for grid in grids:
        b_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(b_graph, stream=capture):
            checked(driver.cuLaunchKernel(function, grid, 1, 1, spec['threads'], 1, 1,
                                          spec['shared_bytes'], capture_driver, args, 0))
        probes[grid] = Executable(driver.CUgraphExec(init_value=b_graph.raw_cuda_graph_exec()), context_id, b_graph)
    producer = driver.CUstream(init_value=torch.cuda.current_stream().cuda_stream)
    prerequisite = checked(driver.cuEventCreate(2))

    class Permit:
        def validate_live(self):
            require_permit(run_dir, scope, worker_path)

    # Verify the actual v2 operations with a condition that is already satisfied.
    flag = checked(driver.cuMemAlloc(4))
    checked(driver.cuMemsetD32Async(flag, 1, 1, producer))
    checked(driver.cuStreamWaitValue32(producer, flag, 1, 0))
    checked(driver.cuStreamWriteValue32(producer, flag, 0, 0))
    checked(driver.cuStreamSynchronize(producer))
    checked(driver.cuMemFree(flag))

    class TaggedBackend(CudaStreamBackend):
        mode = None
        label_launches = True

        def launch(self, executable, stream):
            if profile == 'P2' and self.label_launches:
                role = 'A' if executable is a else 'B'
                with annotate(f'G4|{case}|{block}|{self.mode}|{role}'):
                    super().launch(executable, stream)
            else:
                super().launch(executable, stream)

    backend = TaggedBackend(driver, Permit(), context_id, True, prerequisite)
    modes = MODES[:]
    random.Random(41023 + block).shuffle(modes)

    def run_condition(mode, probe):
        output.fill_(float('nan'))
        destination.fill_(float('nan'))
        checked(driver.cuEventRecord(prerequisite, producer))
        backend.mode = mode
        submitted = submit(backend, mode, a, probe)
        did_a = 'A' in submitted['order']
        did_b = 'B' in submitted['order'] and mode != 'empty_control'
        if did_a and not torch.equal(output, reference):
            raise RuntimeError('QKV output changed under ' + mode)
        if did_b:
            validate_outputs(spec, destination.cpu().numpy())
        return dict(mode=mode, completed_A=int(did_a), completed_B=int(did_b),
                    output_checks_passed=True,
                    joint_event_ns=backend.elapsed_ns('joint_start', 'joint_end'),
                    a_event_ns=backend.elapsed_ns('A_start', 'A_end') if did_a else None,
                    b_event_ns=backend.elapsed_ns('B_start', 'B_end') if did_b else None)

    results = []
    for grid in grids:
        probe = probes[grid]
        backend.label_launches = False
        standalone = run_condition('b_alone', probe) if probe_symbol != 'fp32_light' else None
        backend.label_launches = True
        warmup = [run_condition(mode, probe) for mode in MODES] if profile == 'P0' else []
        records = [run_condition(mode, probe) for mode in modes]
        results.append(dict(scope=f'ONE_{profile}_BLOCK_ISOLATED_CHECKPOINT_PROJECTION', profile=profile, block=block,
                            fp32_probe=dict(spec=spec, grid=grid, cubin_sha256=cubin_sha256,
                                            **work_accounting(spec, grid)), conditions=records, warmup=warmup,
                            probe_identity=dict(symbol=probe_symbol, registers_per_thread=registers,
                                                static_shared_bytes=static_shared, blocks_per_sm=capacity),
                            standalone_acceptance=standalone,
                            overlap_code_sha256=sha(Path(__file__))))
    backend.close_after_join()
    checked(driver.cuEventDestroy(prerequisite))
    for probe in probes.values():
        probe.owner.reset()
    checked(driver.cuModuleUnload(module))
    if compare_grids:
        return dict(profile=profile, block=block, order=grids, same_probe_buffers=True, configurations=results)
    return results[0]
