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


def pilot(torch, driver, run_dir, worker_path, case, graph, output, reference, cubin_sha256,
          profile='P2', block=0):
    if profile == 'P2':
        from nvtx import annotate
    scope = 'standalone_qkv_p0' if profile == 'P0' else 'standalone_qkv_overlap'

    uri = SPATIAL / 'build/offline_sm120_v2/probes_sm120.cubin'
    if sha(uri) != cubin_sha256:
        raise RuntimeError('calibrated CUBIN hash mismatch')
    spec = dict(probe_specs()[0], items=65536, rounds=8, expected_flops=16_777_216)
    grid = 128
    elements = spec['items'] * 16
    source = (0.25 + (torch.arange(elements, dtype=torch.int32) % 64).float() / 128).to('cuda')
    destination = torch.empty_like(source)
    module = checked(driver.cuModuleLoad(os.fsencode(uri)))
    function = checked(driver.cuModuleGetFunction(module, b'fp32_light'))
    context_id = int(checked(driver.cuCtxGetCurrent()))
    args = ((source.data_ptr(), destination.data_ptr(), spec['items'], spec['rounds']),
            (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32))
    capture = torch.cuda.Stream()
    capture_driver = driver.CUstream(init_value=capture.cuda_stream)
    b_graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(b_graph, stream=capture):
        checked(driver.cuLaunchKernel(function, grid, 1, 1, 32, 1, 1, 0, capture_driver, args, 0))
    a = Executable(driver.CUgraphExec(init_value=graph.raw_cuda_graph_exec()), context_id, graph)
    b = Executable(driver.CUgraphExec(init_value=b_graph.raw_cuda_graph_exec()), context_id, b_graph)
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

        def launch(self, executable, stream):
            if profile == 'P2':
                role = 'A' if executable is a else 'B'
                with annotate(f'G4|{case}|{block}|{self.mode}|{role}'):
                    super().launch(executable, stream)
            else:
                super().launch(executable, stream)

    backend = TaggedBackend(driver, Permit(), context_id, True, prerequisite)
    modes = MODES[:]
    random.Random(41023 + block).shuffle(modes)

    def run_condition(mode):
        output.fill_(float('nan'))
        destination.fill_(float('nan'))
        checked(driver.cuEventRecord(prerequisite, producer))
        backend.mode = mode
        submitted = submit(backend, mode, a, b)
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

    warmup = [run_condition(mode) for mode in MODES] if profile == 'P0' else []
    records = [run_condition(mode) for mode in modes]
    backend.close_after_join()
    checked(driver.cuEventDestroy(prerequisite))
    b_graph.reset()
    checked(driver.cuModuleUnload(module))
    return dict(scope=f'ONE_{profile}_BLOCK_ISOLATED_CHECKPOINT_PROJECTION', profile=profile, block=block,
                fp32_probe=dict(spec=spec, grid=grid, cubin_sha256=cubin_sha256,
                                **work_accounting(spec, grid)), conditions=records, warmup=warmup,
                overlap_code_sha256=sha(Path(__file__)))
