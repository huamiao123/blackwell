"""Build independent full-model Graph executables. Never edit the captured vLLM graph."""
import ctypes
import hashlib
import json
from pathlib import Path

from graph_binding import checked

class TargetGraphs:
    def __init__(self, torch, driver):
        self.d=driver
        self.captures={}
        self.variants={}
        self.info={}
        self.hook=torch.cuda.graphs.register_graph_instantiate_hook(self.capture)

    def nodes(self,g):
        d=self.d
        _,count=checked(d.cuGraphGetNodes(g,0))
        seq,n=checked(d.cuGraphGetNodes(g,count))
        return seq[:n]

    def capture(self,graph):
        d=self.d
        g=d.CUgraph(init_value=graph.raw_cuda_graph())
        source=self.nodes(g)
        base=checked(d.cuGraphClone(g))
        originals={}
        signature={}
        for n in source:
            c=checked(d.cuGraphNodeFindInClone(n,base))
            originals[int(c)]=int(checked(d.cuGraphNodeGetToolsId(n)))
            local=int(checked(d.cuGraphNodeGetLocalId(n)))
            if local!=int(checked(d.cuGraphNodeGetLocalId(c))) or local!=(originals[int(c)] & 0xffffffff):
                raise ValueError('clone changed local node identity')
            if checked(d.cuGraphNodeGetType(n))==d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                par=checked(d.cuGraphKernelNodeGetParams(n))
                name=checked(d.cuFuncGetName(par.func))
                signature[local]=[name.decode() if isinstance(name,bytes) else name,
                                  par.gridDimX,par.gridDimY,par.gridDimZ,par.blockDimX,par.blockDimY,par.blockDimZ,par.sharedMemBytes]
        self.captures[graph]=(base,originals,signature)

    def adjacent(self,node,fn):
        _,_,count=checked(fn(node,0))
        if not count:raise ValueError('expected interior target')
        seq,edge,n=checked(fn(node,count))
        for e in edge[:n]:
            if int(e.type) or int(e.from_port) or int(e.to_port):raise ValueError('non-default graph edge')
        return seq[:n]

    def identify(self,base,model,precision,spatial):
        d=self.d
        refs=json.loads((spatial/'configs/extended_kernel_selection.json').read_bytes())['cases']
        targets={}
        for op in ('o','down'):
            case,=[r for r in refs if r['case_id']==f'{precision}_{op}_proj']
            ref=spatial/'qkv_replay_validation'/f'{op}_reference.json'
            if hashlib.sha256(ref.read_bytes()).hexdigest()!=case['reference_sha256']:raise ValueError('historic reference hash')
            hist,=[r for r in json.loads(ref.read_bytes())[precision]['nodes'] if r['tools_id']==case['historical_tools_id']]
            layer=model.get_submodule(case['module'])
            fields=('weight','weight_scale','alpha') if precision=='nvfp4' else ('weight',)
            needles={key:getattr(layer,key).data_ptr().to_bytes(8,'little') for key in fields}
            found=[]
            for n in self.nodes(base):
                if checked(d.cuGraphNodeGetType(n))!=d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:continue
                params=checked(d.cuGraphKernelNodeGetParams(n))
                if ([params.gridDimX,params.gridDimY,params.gridDimZ]!=case['grid']
                    or [params.blockDimX,params.blockDimY,params.blockDimZ]!=case['block']
                    or params.sharedMemBytes!=case['cta']['dynamic_shared_bytes']):continue
                name=checked(d.cuFuncGetName(params.func))
                if isinstance(name,bytes):name=name.decode()
                if name!=hist['kernel_name']:continue
                _,size=checked(d.cuFuncGetParamInfo(params.func,0))
                if size!=hist['parameter_size'] or not 0<size<=65536 or not params.kernelParams or params.extra:continue
                addr=ctypes.c_void_p.from_address(int(params.kernelParams)).value
                if not addr:raise ValueError('null argument')
                payload=ctypes.string_at(addr,size)
                offsets={key:payload.find(value) for key,value in needles.items()}
                if min(offsets.values())>=0:
                    found.append((n,dict(module=case['module'],symbol=name,grid=case['grid'],block=case['block'],
                                          reference_sha256=case['reference_sha256'],bound_offsets=offsets)))
            if len(found)!=1:raise ValueError(f'{op} expected one bound node, found {len(found)}')
            targets[op]=found[0]
        return targets

    def build(self,graph,hooks,model,precision,spatial,modes):
        d=self.d
        base,originals,self.native_signature=self.captures[graph]
        self.selected=graph
        self.args=[]
        for out in [*hooks.buffers[0],hooks.warm_buffer]:
            ptrs=tuple(t.data_ptr() for t in (*hooks.sources,out))
            params=d.CUDA_KERNEL_NODE_PARAMS()
            params.func=hooks.function
            params.gridDimX,params.gridDimY,params.gridDimZ=64,1,1
            params.blockDimX,params.blockDimY,params.blockDimZ=hooks.spec['threads'],1,1
            params.sharedMemBytes=hooks.spec['shared_bytes']
            cells=[ctypes.c_uint64(p) for p in ptrs]+[ctypes.c_uint32(hooks.tiles),ctypes.c_uint32(hooks.rounds)]
            array=(ctypes.c_void_p*len(cells))(*(ctypes.addressof(c) for c in cells))
            params.kernelParams=ctypes.addressof(array)
            self.args.append((params,cells,array))
        targets=self.identify(base,model,precision,spatial)
        original_count=len(originals)
        self.info['original']=dict(targets={k:dict(**v,tools_id=originals[int(n)]) for k,(n,v) in targets.items()},
                                   nodes=original_count,exec_id=int(checked(d.cuGraphExecGetId(
                                       d.CUgraphExec(init_value=graph.raw_cuda_graph_exec())))))
        upload=d.CUstream(init_value=hooks.torch.cuda.current_stream().cuda_stream)
        for mode in modes:
            if mode=='original':continue
            clone=checked(d.cuGraphClone(base))
            current={k:checked(d.cuGraphNodeFindInClone(n,clone)) for k,(n,_) in targets.items()}
            for k,node in current.items():
                if int(checked(d.cuGraphNodeGetLocalId(node))) != (originals[int(targets[k][0])]&0xffffffff):
                    raise ValueError('target changed local identity')
            b=None
            if mode!='sham':
                op,order=mode.split('_')
                a=current[op]
                preds=self.adjacent(a,d.cuGraphNodeGetDependencies)
                succ=self.adjacent(a,d.cuGraphNodeGetDependentNodes)
                deps=[a] if order=='ab' else preds
                b=checked(d.cuGraphAddKernelNode(clone,deps,len(deps),self.args[0][0]))
                after=[a] if order=='ba' else succ
                checked(d.cuGraphAddDependencies(clone,[b]*len(after),after,None,len(after)))
            exe=checked(d.cuGraphInstantiate(clone,0))
            self.variants[mode]=(clone,exe,b)
            checked(d.cuGraphUpload(exe,upload))
            count=len(self.nodes(clone))
            if count!=original_count+int(b is not None):raise ValueError('model topology count changed')
            self.info[mode]=dict(nodes=count,exec_id=int(checked(d.cuGraphExecGetId(exe))),
                                targets={k:dict(**targets[k][1],tools_id=int(checked(d.cuGraphNodeGetToolsId(a)))) for k,a in current.items()},
                                probe_tools_id=int(checked(d.cuGraphNodeGetToolsId(b))) if b is not None else None)
        hooks.torch.cuda.current_stream().synchronize()

    def launch(self,mode,slot,stream,runtime):
        g,exe,b=self.variants[mode]
        if b is not None:
            checked(self.d.cuGraphExecKernelNodeSetParams(exe,b,self.args[slot][0]))
        checked(runtime.cudaGraphLaunch(int(exe),stream.cuda_stream))

    def close(self):
        for clone,exe,_ in self.variants.values():
            checked(self.d.cuGraphExecDestroy(exe))
            checked(self.d.cuGraphDestroy(clone))
        for base,_,_ in self.captures.values():checked(self.d.cuGraphDestroy(base))
