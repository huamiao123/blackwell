"""Read-only live-graph metadata, parameter references, and exact tensor matching.

No CUDA calls occur at import. Parameter bytes are CPU-owned API results,
read within the graph lifetime and only at sizes returned by the driver.
"""
import ctypes
import hashlib
import weakref


ROLES = {'qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'}


def match_references(payload, tensors):
    matches = []
    for tensor in tensors:
        address = tensor['address']
        if not address:
            continue
        needle = address.to_bytes(8, 'little')
        at = payload.find(needle)
        if at >= 0:
            matches.append(dict(module=tensor['module'], tensor=tensor['tensor'], byte_offset=at))
    owners = sorted({m['module'] for m in matches})
    return dict(status='EXACT_UNIQUE_MODULE_REFERENCE' if len(owners) == 1 else
                'AMBIGUOUS_MODULE_REFERENCES' if owners else 'NO_EXACT_REFERENCE',
                module=owners[0] if len(owners) == 1 else None, references=matches)


def tensor_catalog(model, torch):
    result = []
    for name, module in model.named_modules():
        if name.split('.')[-1] not in ROLES:
            continue
        values = dict(module.named_parameters(recurse=False))
        values.update(dict(module.named_buffers(recurse=False)))
        for attr in ['weight', 'weight_scale', 'alpha', 'input_global_scale_inv', 'input_scale', 'weight_scale_2']:
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor):
                values[attr] = value
        for attr, value in values.items():
            if value.is_cuda:
                result.append(dict(module=name, role=name.split('.')[-1], tensor=attr,
                    address=value.data_ptr(), shape=list(value.shape), dtype=str(value.dtype),
                    numel=value.numel(), element_size=value.element_size()))
    return result


def checked(result):
    if int(result[0]) != 0:
        raise RuntimeError('CUDA metadata API returned ' + str(result[0]))
    return result[1:] if len(result) != 2 else result[1]


class GraphObserver:
    def __init__(self, torch):
        from cuda.bindings import driver, runtime
        self.driver, self.runtime = driver, runtime
        self.snapshots = []
        self.live = []
        self.replays = []
        self.handle = torch.cuda.graphs.register_graph_instantiate_hook(self.observe)

    def observe(self, graph):
        d, r = self.driver, self.runtime
        # The default graph is still live inside instantiate hooks. No debug
        # retention, re-instantiation, graph mutation or replay hook is enabled.
        topology = graph.get_graph_data()
        raw = graph.raw_cuda_graph()
        _, count = checked(r.cudaGraphGetNodes(raw, numNodes=0))
        nodes, count = checked(r.cudaGraphGetNodes(raw, numNodes=count))
        by_node = {n['node_id']: n for n in topology['nodes']}
        kernels = []
        for node in nodes[:count]:
            if checked(r.cudaGraphNodeGetType(node)) != r.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                continue
            tools = checked(r.cudaGraphNodeGetToolsId(node))
            info = by_node[tools & 0xffffffff]
            params = checked(d.cuGraphKernelNodeGetParams(d.CUgraphNode(init_value=int(node))))
            name = checked(d.cuFuncGetName(params.func))
            if isinstance(name, bytes):
                name = name.decode()
            if 'sm120' not in name.lower() or 'e2m1' not in name.lower():
                continue
            row = dict(node_id=info['node_id'], graph_id=info['graph_id'], tools_id=info['tools_id'],
                       kernel_name=name, dependencies=info['dependencies'],
                       grid=[params.gridDimX, params.gridDimY, params.gridDimZ],
                       block=[params.blockDimX, params.blockDimY, params.blockDimZ], shared_bytes=params.sharedMemBytes)
            offset, size = checked(d.cuFuncGetParamInfo(params.func, 0))
            row.update(parameter_index=0, parameter_offset=offset, parameter_size=size)
            if params.kernelParams and not params.extra and 0 < size <= 65536:
                args_ptr = int(params.kernelParams)
                value_ptr = ctypes.c_void_p.from_address(args_ptr).value
                if not value_ptr:
                    raise RuntimeError('null host kernel parameter value')
                payload = ctypes.string_at(value_ptr, size)
                row.update(parameter_bytes_hex=payload.hex(), parameter_sha256=hashlib.sha256(payload).hexdigest(),
                           parameter_read_status='DRIVER_SIZED_CPU_OWNED_ARGUMENT_COPY')
            else:
                row['parameter_read_status'] = 'UNAVAILABLE_EXTRA_BUFFER_OR_UNSUPPORTED_SIZE'
            kernels.append(row)
        index = len(self.snapshots)
        self.live.append((weakref.ref(graph), index))
        self.snapshots.append(dict(capture_index=index, topology=topology, native_kernels=kernels))

    def replay(self, graph, iteration):
        matches = [index for ref, index in self.live if ref() is graph]
        if len(matches) != 1:
            raise RuntimeError('missing/ambiguous graph capture identity')
        self.replays.append(dict(iteration=iteration, capture_index=matches[0],
                                 exec_graph_id=self.snapshots[matches[0]]['topology']['exec_graph_id']))

    def bind(self, tensors):
        for capture in self.snapshots:
            for node in capture['native_kernels']:
                node['binding'] = match_references(bytes.fromhex(node.get('parameter_bytes_hex', '')), tensors)

    def close(self):
        self.handle.remove()
