"""BF16 WMMA control; immutable reference-matching methods reused from Gate-3 NVFP4."""
from pathlib import Path
import sys,ctypes,hashlib,weakref
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate3_backbone/src'))
from graph_binding import GraphObserver,checked,match_references

class BF16Observer(GraphObserver):
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
            if 'wmma' not in name.lower() or 'bf16' not in name.lower():
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
        self.snapshots.append(dict(capture_index=index, topology=topology, matrix_kernels=kernels))

    def bind(self, tensors):
        for capture in self.snapshots:
            for node in capture['matrix_kernels']:
                node['binding'] = match_references(bytes.fromhex(node.get('parameter_bytes_hex', '')), tensors)
