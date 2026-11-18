"""Validate one exact NVFP4 projection with a private single-kernel Graph."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path

from g4_common import SPATIAL, V4, load, save, sha
from validate_probe_gpu import require_permit


def read_projection(model, module_name, expected_keys, suffixes):
    from safetensors import safe_open

    index = model / 'model.safetensors.index.json'
    layouts = load(index)['weight_map']
    prefix = module_name.removesuffix('qkv_proj')
    names = [f'{prefix}{role}_proj.{suffix}' for suffix in suffixes for role in ('q', 'k', 'v')]
    if not set(names).issubset(expected_keys):
        raise ValueError('checkpoint field list changed')
    sources = {name: layouts[name] for name in names}
    if not all(Path(name).name == name and name.endswith('.safetensors') for name in sources.values()):
        raise ValueError('unexpected local shard name')
    with ExitStack() as stack:
        cursor = {name: stack.enter_context(safe_open(model / name, framework='pt', device='cpu'))
                  for name in sorted(set(sources.values()))}
        params = {field: [cursor[sources[name]].get_tensor(name)
                          for name in (f'{prefix}{role}_proj.{field}' for role in ('q', 'k', 'v'))]
                  for field in suffixes}
    return params, sources, sha(index)


def load_layer(torch, model, module_name, expected_keys):
    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel

    params, sources, index_sha = read_projection(model, module_name, expected_keys,
                                                  ['weight', 'weight_scale', 'weight_scale_2', 'input_scale'])

    def concatenate(tensors):
        dtype = tensors[0].dtype
        if any(tensor.dtype != dtype for tensor in tensors):
            raise ValueError('inconsistent projection dtype')
        # Concatenate one-byte scale bit patterns. CPU FP8 cat is version-dependent.
        if dtype == torch.float8_e4m3fn:
            return torch.cat([tensor.view(torch.uint8) for tensor in tensors]).view(dtype).to('cuda')
        return torch.cat(tensors).to('cuda')

    layer = torch.nn.Module()
    for field in ('weight', 'weight_scale'):
        layer.register_parameter(field, torch.nn.Parameter(concatenate(params[field]), requires_grad=False))
    for field, source in [('input_global_scale', 'input_scale'), ('weight_global_scale', 'weight_scale_2')]:
        if any(tensor.numel() != 1 for tensor in params[source]):
            raise ValueError('expected a global scale per Q/K/V shard')
        scale = torch.stack([tensor.reshape(()) for tensor in params[source]]).max().to('cuda', torch.float32)
        layer.register_parameter(field, torch.nn.Parameter(scale, requires_grad=False))
    layer.input_global_scale_inv = torch.nn.Parameter(1 / layer.input_global_scale, requires_grad=False)
    layer.alpha = torch.nn.Parameter(layer.input_global_scale * layer.weight_global_scale, requires_grad=False)
    layer.output_size_per_partition = layer.weight.shape[0]
    layer.input_size_per_partition = layer.weight.shape[1] * 2
    backend = CutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
    backend.process_weights_after_loading(layer)
    return layer, backend, sources, index_sha


def exact_graph_node(graph, driver, runtime, selected, expected, layer):
    sys.path.insert(0, str(V4 / 'gate3_backbone/src'))
    from graph_binding import checked

    raw = graph.raw_cuda_graph()
    _, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=0))
    nodes, count = checked(runtime.cudaGraphGetNodes(raw, numNodes=count))
    if count != 1 or checked(runtime.cudaGraphNodeGetType(nodes[0])) != runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
        raise RuntimeError(f'expected one kernel without any upstream graph nodes, observed {count} graph nodes')
    node = nodes[0]
    tools_id = checked(runtime.cudaGraphNodeGetToolsId(node))
    params = checked(driver.cuGraphKernelNodeGetParams(driver.CUgraphNode(init_value=int(node))))
    name = checked(driver.cuFuncGetName(params.func))
    if isinstance(name, bytes):
        name = name.decode()
    grid = [params.gridDimX, params.gridDimY, params.gridDimZ]
    block = [params.blockDimX, params.blockDimY, params.blockDimZ]
    if (name != expected['kernel_name'] or grid != selected['grid'] or block != selected['block'] or
            params.sharedMemBytes != selected['cta']['dynamic_shared_bytes']):
        raise RuntimeError(f'Gate 3 kernel identity differs: {name}, {grid=}, {block=}, {params.sharedMemBytes=}')
    offset, size = checked(driver.cuFuncGetParamInfo(params.func, 0))
    if not params.kernelParams or params.extra or size != expected['parameter_size']:
        raise RuntimeError('unexpected parameter encoding')
    source = ctypes.c_void_p.from_address(int(params.kernelParams)).value
    if not source:
        raise RuntimeError('invalid Graph API parameter reference')
    data = ctypes.string_at(source, size)
    fields = ('weight', 'weight_scale', 'alpha') if selected['precision'] == 'nvfp4' else ('weight',)
    references = {field: data.find(getattr(layer, field).data_ptr().to_bytes(8, 'little')) for field in fields}
    if any(found < 0 for found in references.values()):
        raise RuntimeError('live layer-18 weights or scales do not bind to the captured node')
    attr = driver.CUfunction_attribute
    attrs = {name: checked(driver.cuFuncGetAttribute(getattr(attr, 'CU_FUNC_ATTRIBUTE_' + name), params.func))
             for name in ('NUM_REGS', 'SHARED_SIZE_BYTES', 'MAX_DYNAMIC_SHARED_SIZE_BYTES')}
    if attrs['NUM_REGS'] != selected['cta']['registers_per_thread'] or attrs['SHARED_SIZE_BYTES'] != selected['cta']['static_shared_bytes']:
        raise RuntimeError('kernel resources differ from Gate 3')
    blocks = checked(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(params.func, block[0], params.sharedMemBytes))
    return dict(kernel_name=name, grid=grid, block=block, dynamic_shared_bytes=params.sharedMemBytes,
                register_and_shared_attributes=attrs, homogeneous_api_blocks_per_sm=blocks,
                tools_id=int(tools_id), parameter_offset=offset, parameter_size=size, live_reference_offsets=references)


def tensor_hash(torch, tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--precision', choices=['nvfp4', 'bf16'], default='nvfp4')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--p0-block', type=int, choices=range(6))
    args = parser.parse_args()
    p0 = args.p0_block is not None
    if p0 and not args.overlap:
        raise ValueError('P0 block requires an overlap experiment')
    scope = 'standalone_qkv_p0' if p0 else 'standalone_qkv_overlap' if args.overlap else 'standalone_qkv_replay'
    permit = require_permit(args.run_dir, scope, Path(__file__))
    candidates = load(SPATIAL / 'configs/kernel_selection.json')['cases']
    precision = args.precision
    selected = next(row for row in candidates if row['case_id'] == precision + '_qkv_proj')
    historical = load(SPATIAL / 'qkv_replay_validation/gate3_reference.json')[precision]
    expected = next(row for row in historical['nodes'] if row['tools_id'] == selected['historical_tools_id'])

    import torch
    from cuda.bindings import driver, runtime
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import nvfp4_weight_padding_bytes
    from vllm.model_executor.kernels.linear.nvfp4 import cutlass
    from vllm.model_executor.layers import utils as linear_utils

    torch.cuda.set_device(0)
    status, device = driver.cuDeviceGet(0)
    if int(status):
        raise RuntimeError('cannot identify GPU')
    status, device_uuid = driver.cuDeviceGetUuid(device)
    if int(status) or str(uuid.UUID(bytes=bytes(device_uuid.bytes))) != permit['gpu_uuid'].removeprefix('GPU-'):
        raise RuntimeError('GPU UUID mismatch')

    with torch.inference_mode():
        model_root = Path.home() / 'fp4-llm/models'
        if precision == 'nvfp4':
            model = model_root / 'Qwen3-8B-NVFP4'
            layer, backend, sources, index_sha = load_layer(torch, model, selected['module'], historical['checkpoint_keys'])
            width = layer.input_size_per_partition
            backend_path = cutlass.__file__
        else:
            model = model_root / 'Qwen3-8B-BF16'
            fields, sources, index_sha = read_projection(model, selected['module'], historical['checkpoint_keys'], ['weight'])
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(torch.cat(fields['weight']).to('cuda'), requires_grad=False)
            width = layer.weight.shape[1]
            backend_path = linear_utils.__file__
        for item in historical['tensors']:
            tensor = getattr(layer, item['tensor'])
            if list(tensor.shape) != item['shape'] or str(tensor.dtype) != item['dtype']:
                raise RuntimeError('processed layer tensor differs from Gate 3 layout: ' + item['tensor'])
        x = (((torch.arange(8 * width, dtype=torch.int32) * 31) % 509 - 254).reshape(8, width)
             .to(torch.float32) / 1024).to('cuda', torch.bfloat16)
        if precision == 'nvfp4':
            reference = backend.apply_weights(layer, x).clone()
            packed, scales = scaled_fp4_quant(x, layer.input_global_scale_inv, is_sf_swizzled_layout=True,
                                              backend='cutlass', padded_n=width + 2 * nvfp4_weight_padding_bytes(layer))

            def project():
                return cutlass_scaled_fp4_mm(packed, layer.weight, scales, layer.weight_scale, layer.alpha, x.dtype)

            retained = (packed, scales)
        else:
            unquantized = linear_utils.dispatch_unquantized_gemm('cutlass')

            def project():
                return unquantized(layer, x, layer.weight, None)

            reference = project().clone()
            retained = (x,)
        if not torch.isfinite(reference).all().item() or not torch.count_nonzero(reference).item():
            raise RuntimeError('invalid reference output')
        if not torch.equal(reference, project()):
            raise RuntimeError('direct call differs from original projection')
        before = [tensor_hash(torch, tensor) for tensor in retained]
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            output = project()
        identity = exact_graph_node(graph, driver, runtime, selected, expected, layer)
        graph.instantiate()
        observations = []
        for trial in range(3):
            # Retain graph, packed inputs, scales, weights and output for all launches.
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(reference, output):
                raise RuntimeError('single-node replay did not reproduce the original projected output')
            observations.append(dict(trial=trial, exact_match=True, output_sha256=tensor_hash(torch, output)))
        if before != [tensor_hash(torch, tensor) for tensor in retained]:
            raise RuntimeError('projection mutated its input buffers')
        result = dict(status='QKV_REPLAY_OUTPUT_VERIFIED', scope='ISOLATED_CHECKPOINT_QKV_SYNTHETIC_ACTIVATION',
                      case=selected['case_id'], module=selected['module'], live_identity=identity,
                      source_weights=sources, index_sha256=index_sha, input_shape=list(x.shape),
                      input_sha256=tensor_hash(torch, x), input_is_synthetic=True, output_shape=list(output.shape),
                      trials=observations, torch_version=torch.__version__,
                      vllm_version=importlib.metadata.version('vllm'),
                      backend_source_sha256=sha(Path(backend_path)),
                      gate3_reference_sha256=sha(SPATIAL / 'qkv_replay_validation/gate3_reference.json'),
                      full_model_loaded=False, metrics_scope='correctness, identity, and function attributes only')
        if args.overlap:
            from qkv_overlap_pilot import pilot
            result['overlap_pilot'] = pilot(torch, driver, args.run_dir, Path(__file__), selected['case_id'],
                                            graph, output, reference, permit['cubin_sha256'],
                                            profile='P0' if p0 else 'P2', block=args.p0_block if p0 else 0)
            result['status'] = 'QKV_OVERLAP_PILOT_OUTPUTS_VERIFIED'
        save(args.run_dir / 'result.json', result)
        print(result['status'], f'{len(observations)} exact output matches', flush=True)


if __name__ == '__main__':
    main()
