"""Live-node cloning only. Historical pointers are never accepted as a replay source."""
from contracts import ContractError

def checked(result):
    if int(result[0])!=0:raise RuntimeError('CUDA API failed: '+str(result[0]))
    return result[1] if len(result)==2 else result[1:]

class DiagnosticNode:
    def __init__(self,driver,graph,node,lease,context_identity):
        self.driver,self.graph,self.node=driver,graph,node
        self.lease,self.identity=lease,context_identity
        self.executable=None

    @classmethod
    def copy_during_live_capture(cls,driver,source_node,lease,context_identity):
        # Caller must have an authorized CUDA context and call this while the
        # source graph template is alive. AddKernelNode copies argument values.
        if lease.closed or lease.graph_owner is None or lease.model_owner is None:
            raise ContractError('missing live owners')
        if (lease.process_id,lease.context_id,lease.device_uuid)!=tuple(context_identity):
            raise ContractError('not the current process/context/device')
        params=checked(driver.cuGraphKernelNodeGetParams(source_node))
        graph=checked(driver.cuGraphCreate(0))
        try:
            node=checked(driver.cuGraphAddKernelNode(graph,None,0,params))
            checked(driver.cuGraphKernelNodeCopyAttributes(node,source_node))
        except BaseException:
            checked(driver.cuGraphDestroy(graph));raise
        # This creates no workload and does not assert that input data is ready.
        return cls(driver,graph,node,lease,context_identity)

    def instantiate(self):
        self.lease.validate(*self.identity)
        if self.executable is not None:raise ContractError('already instantiated')
        self.executable=checked(self.driver.cuGraphInstantiate(self.graph,0))
        return self.executable

    def launch(self,stream):
        self.lease.validate(*self.identity)
        if self.executable is None:raise ContractError('no validated executable')
        checked(self.driver.cuGraphLaunch(self.executable,stream))

    def close_after_join(self,join_completed):
        if not join_completed:raise ContractError('cannot free in-flight graph/buffers')
        if self.executable is not None:
            checked(self.driver.cuGraphExecDestroy(self.executable));self.executable=None
        if self.graph is not None:
            checked(self.driver.cuGraphDestroy(self.graph));self.graph=None
        self.lease.closed=True

def ancestor_closure(nodes,target_index):
    """CPU plan for a *separate clone*; never prune the original graph."""
    by_index={n['index']:n for n in nodes}
    if target_index not in by_index:raise ContractError('unknown target')
    seen=set();visiting=set()
    def visit(i):
        if i in visiting:raise ContractError('cycle in graph metadata')
        if i in seen:return
        if i not in by_index:raise ContractError('missing predecessor')
        visiting.add(i)
        for dep in by_index[i]['dependencies']:visit(dep)
        visiting.remove(i);seen.add(i)
    visit(target_index)
    unsupported=[i for i in seen if by_index[i]['node_type'] not in ['kernel','empty']]
    return dict(ancestor_indices=sorted(seen-{target_index}),target_index=target_index,
                requires_external_dependency_review=bool(unsupported),unsupported_nodes=unsupported,
                status='CPU_PLAN_ONLY_NOT_A_READY_INPUT_PROOF')
