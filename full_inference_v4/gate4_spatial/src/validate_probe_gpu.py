"""Standalone probe acceptance with live attributes and full output checks."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

from g4_common import SPATIAL, load, save, sha
from plan import probe_specs
from probe_reference import validate_outputs, work_accounting


def require_permit(run_dir):
    permit = load(run_dir / 'permit.json')
    if (permit['scope'] != 'probe_only' or
            permit['nonce'] != os.environ.get('G4_PROBE_NONCE') or
            permit['supervisor_pid'] != os.getppid() or
            not time.time() < permit['deadline'] or
            permit['worker_sha256'] != sha(Path(__file__))):
        raise RuntimeError('no valid supervisor permit for standalone probe')
    return permit


def run_spec(driver, call, np, module, stream, spec, grids, result, checkpoint):
    is_fma = spec['probe'] == 'fp32_fma'
    prefix = 'fp32_' if is_fma else 'tensor_'
    symbol = prefix + spec['variant']
    function = call(driver.cuModuleGetFunction, module, symbol.encode())
    attributes = driver.CUfunction_attribute
    resource_names = ('NUM_REGS', 'SHARED_SIZE_BYTES', 'MAX_THREADS_PER_BLOCK',
                      'MAX_DYNAMIC_SHARED_SIZE_BYTES', 'LOCAL_SIZE_BYTES',
                      'PREFERRED_SHARED_MEMORY_CARVEOUT')
    resources = {name: call(driver.cuFuncGetAttribute, getattr(attributes, 'CU_FUNC_ATTRIBUTE_' + name), function)
                 for name in resource_names}
    occupancy = call(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor,
                     function, spec['threads'], spec['shared_bytes'])

    if is_fma:
        count = spec['items'] * spec['vector']
        pattern = np.float32(0.25) + np.arange(64, dtype=np.float32) / np.float32(128)
        inputs = [np.tile(pattern, count // 64)]
        work_size, rounds = spec['items'], spec['rounds']
    else:
        count = spec['tiles'] * spec['m'] * spec['n']
        outer = np.arange(16)[:, None]
        inner = np.arange(spec['k_staging'])[None, :]
        pattern = (outer + inner) % 4
        # The BF16 values are exact binary fractions of the FP32 values.
        inputs = [((base + step * pattern).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
                  for base, step in [(0.5, 0.125), (0.25, 0.0625)]]
        work_size, rounds = spec['tiles'], spec['mma_rounds']

    output = np.empty(count, dtype=np.float32)
    input_buffers = []
    for array in inputs:
        device_buffer = call(driver.cuMemAlloc, array.nbytes)
        call(driver.cuMemcpyHtoD, device_buffer, array.ctypes.data, array.nbytes)
        input_buffers.append(device_buffer)
    out_ptr = call(driver.cuMemAlloc, output.nbytes)
    begin = call(driver.cuEventCreate, 0)
    end = call(driver.cuEventCreate, 0)
    arguments = (tuple(int(p) for p in [*input_buffers, out_ptr]) + (work_size, rounds),
                 (ctypes.c_uint64,) * (1 + len(input_buffers)) + (ctypes.c_uint32, ctypes.c_uint32))

    for grid in grids:
        samples, hashes = [], []
        for trial in range(6):
            # Poison all outputs before each launch so stale data cannot pass.
            call(driver.cuMemsetD32Async, out_ptr, 0x7fc00000, count, stream)
            call(driver.cuEventRecord, begin, stream)
            call(driver.cuLaunchKernel, function, grid, 1, 1, spec['threads'], 1, 1,
                 spec['shared_bytes'], stream, arguments, 0)
            call(driver.cuEventRecord, end, stream)
            call(driver.cuEventSynchronize, end)
            elapsed_us = call(driver.cuEventElapsedTime, begin, end) * 1000
            call(driver.cuMemcpyDtoH, output.ctypes.data, out_ptr, output.nbytes)
            evidence = validate_outputs(spec, output)
            if elapsed_us <= 0:
                raise RuntimeError('invalid standalone timing')
            if trial:
                samples.append(elapsed_us)
                hashes.append(hashlib.sha256(output).hexdigest())
        if len(set(hashes)) != 1:
            raise RuntimeError('standalone output varies across repeats')
        result['rows'].append(dict(symbol=symbol, grid=grid, threads=spec['threads'], spec=spec,
                                   dynamic_shared_bytes=spec['shared_bytes'], attributes=resources,
                                   homogeneous_blocks_per_sm=occupancy, **work_accounting(spec, grid),
                                   **evidence, output_sha256=hashes[0], elapsed_us=samples,
                                   timing='one warmup plus five CUDA event intervals; output copies outside intervals'))
        checkpoint()

    for event in [begin, end]:
        call(driver.cuEventDestroy, event)
    for buffer in [*input_buffers, out_ptr]:
        call(driver.cuMemFree, buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--mode', choices=['acceptance', 'qkv_calibration'], default='acceptance')
    args = parser.parse_args()
    permit = require_permit(args.run_dir)
    cubin = SPATIAL / 'build/offline_sm120_v2/probes_sm120.cubin'
    if sha(cubin) != permit['cubin_sha256']:
        raise RuntimeError('CUBIN hash differs from compiled evidence')

    import numpy as np
    from cuda.bindings import driver

    def call(api, *values):
        status, *returned = api(*values)
        if int(status):
            raise RuntimeError(f'{api.__name__}: {status!r}')
        if returned:
            return returned[0] if len(returned) == 1 else returned

    call(driver.cuInit, 0)
    device = call(driver.cuDeviceGet, 0)
    import uuid
    device_uuid = str(uuid.UUID(bytes=bytes(call(driver.cuDeviceGetUuid, device).bytes)))
    if 'GPU-' + device_uuid != permit['gpu_uuid']:
        raise RuntimeError('worker CUDA device does not match supervisor device UUID')
    context = call(driver.cuDevicePrimaryCtxRetain, device)
    call(driver.cuCtxSetCurrent, context)
    module = call(driver.cuModuleLoad, os.fsencode(cubin))
    stream = call(driver.cuStreamCreate, 1)
    attrs = driver.CUdevice_attribute
    keys = ['MULTIPROCESSOR_COUNT', 'COMPUTE_CAPABILITY_MAJOR', 'COMPUTE_CAPABILITY_MINOR',
            'MAX_REGISTERS_PER_MULTIPROCESSOR', 'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
            'RESERVED_SHARED_MEMORY_PER_BLOCK', 'CONCURRENT_KERNELS', 'CAN_USE_STREAM_MEM_OPS']
    device_info = {}
    for name in keys:
        enum = getattr(attrs, 'CU_DEVICE_ATTRIBUTE_' + name, None)
        if enum is None:
            device_info[name] = 'UNAVAILABLE_IN_BINDING'
        else:
            status, value = driver.cuDeviceGetAttribute(enum, device)
            device_info[name] = value if int(status) == 0 else repr(status)
    result = dict(status='MEASURING', scope='STANDALONE_PROBE_ONLY', gpu_uuid=device_uuid,
                  model_weights_loaded=False, driver_api_version=call(driver.cuDriverGetVersion),
                  cuda_bindings_version=importlib.metadata.version('cuda-bindings'),
                  device_attributes=device_info, cubin_sha256=permit['cubin_sha256'],
                  mode=args.mode, rows=[])

    def checkpoint():
        temp = args.run_dir / 'result.tmp'
        save(temp, result)
        temp.replace(args.run_dir / 'result.json')

    checkpoint()
    specs = probe_specs()
    if args.mode == 'qkv_calibration':
        # 512 items per logical group. 65536 items keep all 128 CTAs nonempty.
        # At each round count, exactly the same workload runs at both grids.
        specs = [dict(specs[0], items=65536, rounds=rounds,
                      expected_flops=65536 * 16 * rounds * 2) for rounds in (8, 32)]
    for spec in specs:
        run_spec(driver, call, np, module, stream, spec, [32, 128], result, checkpoint)
    call(driver.cuStreamDestroy, stream)
    call(driver.cuModuleUnload, module)
    call(driver.cuCtxSetCurrent, driver.CUcontext(0))
    call(driver.cuDevicePrimaryCtxRelease, device)
    result['status'] = 'STANDALONE_PROBE_GPU_VERIFIED'
    checkpoint()


if __name__ == '__main__':
    main()
