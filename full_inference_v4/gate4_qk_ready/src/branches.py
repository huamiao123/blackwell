"""Move only dependency edges between already captured Q/K producers."""
import ctypes
from graph_binding import checked

PREFIX = 'triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_'

def args(d, params, count):
    """Read driver-sized CPU copies of pointer/scalar parameters, never device memory."""
    values = []
    if not params.kernelParams or params.extra:
        raise ValueError('unsupported argument representation')
    slots = ctypes.cast(int(params.kernelParams), ctypes.POINTER(ctypes.c_void_p))
    for index in range(count):
        result = d.cuFuncGetParamInfo(params.func, index)
        if int(result[0]) != 0:
            break
        size = result[-1]
        if size <= 8:
            values.append(int.from_bytes(ctypes.string_at(slots[index], size), 'little'))
        else:
            values.append(None)
    return values

def disjoint(a, b):
    return a[0] + a[1] <= b[0] or b[0] + b[1] <= a[0]

def adjacent(d, node, fn, retained_edge=None):
    _, _, count = checked(fn(node, 0))
    if count == 0:
        return []
    mapped, edges, n = checked(fn(node, count))
    kinds = [(int(e.type),int(e.from_port),int(e.to_port)) for e in edges[:n]]
    if retained_edge is not None:
        retained_edge.extend(kinds)
    elif any(any(kind) for kind in kinds):
        raise ValueError('unsupported dependency kind')
    return list(mapped[:n])

class Branches:
    def __init__(self, torch, d):
        self.torch = torch
        self.d = d
        self.captures = {}
        self.executables = {}
        self.hook = torch.cuda.graphs.register_graph_instantiate_hook(self.capture)

    def capture(self, graph):
        d = self.d
        source = d.CUgraph(init_value=graph.raw_cuda_graph())
        self.captures[graph] = checked(d.cuGraphClone(source))

    def build(self, graph, model):
        d = self.d
        self.graph = graph
        self.hook.remove()
        base = self.captures[graph]
        nodes, count = checked(d.cuGraphGetNodes(base, checked(d.cuGraphGetNodes(base, 0))[-1]))
        data = {}
        for node in nodes[:count]:
            if checked(d.cuGraphNodeGetType(node)) != d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                continue
            p = checked(d.cuGraphKernelNodeGetParams(node))
            name = checked(d.cuFuncGetName(p.func)).decode()
            data[int(node)] = (node, p, name)
        by_weight = {}
        for name, m in model.named_modules():
            if name.endswith('.self_attn.q_norm') or name.endswith('.self_attn.k_norm'):
                by_weight[m.weight.data_ptr()] = (int(name.split('.layers.')[1].split('.')[0]), name.split('.')[-1])
        tables = {t.data_ptr(): t.numel() * t.element_size() for name, t in model.named_buffers() if 'cos_sin_cache' in name}
        groups, records = [], []
        for q, qp, qname in data.values():
            if not qname.startswith(PREFIX) or qp.gridDimX != 32:
                continue
            def one(n, f):
                single, = adjacent(d, n, f)
                return data[int(single)]
            pred, _, pred_name = one(q, d.cuGraphNodeGetDependencies)
            k, kp, kname = one(q, d.cuGraphNodeGetDependentNodes)
            cache, cp, cache_name = one(k, d.cuGraphNodeGetDependentNodes)
            cache_edge = []
            att_handle, = adjacent(d,cache,d.cuGraphNodeGetDependentNodes,cache_edge)
            att, ap, att_name = data[int(att_handle)]
            if not ('cutlass' in pred_name and kname.startswith(PREFIX) and kp.gridDimX == 8 and
                    'reshape_and_cache_flash_kernel' in cache_name and 'kernel_mha' in att_name):
                raise ValueError('unexpected producer/consumer chain')
            if int(one(k, d.cuGraphNodeGetDependencies)[0]) != int(q) or int(one(cache,d.cuGraphNodeGetDependencies)[0]) != int(k):
                raise ValueError('unexpected incoming edges')
            qa, ka = args(d, qp, 5), args(d, kp, 5)
            if len(qa) != 5 or len(ka) != 5 or not all(qa + ka):
                raise ValueError('Q/K ABI changed')
            layer, qr = by_weight[qa[1]]
            if by_weight[ka[1]] != (layer,'k_norm') or qr != 'q_norm' or any(qa[i] != ka[i] for i in (0,2,3)):
                raise ValueError('Q/K input bindings changed')
            inputs = [(qa[0],8*6144*2),(qa[1],128*2),(ka[1],128*2),(qa[2],8*8),(qa[3],tables[qa[3]])]
            outputs = [(qa[4],8*32*128*2),(ka[4],8*8*128*2)]
            if not disjoint(*outputs) or not all(disjoint(out, inp) for out in outputs for inp in inputs):
                raise ValueError(f'unsafe tensor alias at layer {layer}')
            kv_args, att_args = args(d, cp, 12), args(d, ap, 20)
            if kv_args[0] != ka[4] or att_args.count(qa[4]) != 1:
                raise ValueError(f'cannot verify real K/Q consumers at layer {layer}')
            ids = [int(checked(d.cuGraphNodeGetLocalId(n))) for n in [pred,q,k,cache,att]]
            groups.append((pred,q,k,cache))
            records.append(dict(layer=layer,nodes=ids,q_name=qname,k_name=kname,producer=pred_name,
                                kv_consumer=cache_name,q_consumer=att_name,k_cache_arg=0,
                                q_attention_arg=att_args.index(qa[4]),retained_cache_edge=cache_edge,
                                disjoint_outputs_and_inputs=True))
        if sorted(row['layer'] for row in records) != list(range(36)):
            raise ValueError(f'expected 36 recognized Q/K pairs, found {len(records)}')
        self.records = sorted(records, key=lambda r:r['layer'])
        self.native_exec = int(checked(d.cuGraphExecGetId(d.CUgraphExec(init_value=graph.raw_cuda_graph_exec()))))
        self.kernel_count = len(data)
        for mode in ('sham','fork'):
            clone = checked(d.cuGraphClone(base))
            if mode == 'fork':
                for group in groups:
                    pred,q,k,cache = [checked(d.cuGraphNodeFindInClone(n,clone)) for n in group]
                    checked(d.cuGraphAddDependencies(clone,[pred,q],[k,cache],None,2))
                    checked(d.cuGraphRemoveDependencies(clone,[q],[k],None,1))
            exe = checked(d.cuGraphInstantiate(clone,0))
            stream = d.CUstream(init_value=self.torch.cuda.current_stream().cuda_stream)
            checked(d.cuGraphUpload(exe,stream))
            self.executables[mode] = (clone,exe)
        self.torch.cuda.current_stream().synchronize()

    def close(self):
        for clone, exe in self.executables.values():
            checked(self.d.cuGraphExecDestroy(exe))
            checked(self.d.cuGraphDestroy(clone))
        for g in self.captures.values():
            checked(self.d.cuGraphDestroy(g))
