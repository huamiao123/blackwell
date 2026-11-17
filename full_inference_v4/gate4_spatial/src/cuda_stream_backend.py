"""Deferred CUDA adapter for stream_protocol. Construction requires a live permit.

No CUDA import/call occurs at module import. GPU integration is not validated.
"""
from dataclasses import dataclass
from replay import checked
from contracts import ContractError

@dataclass
class Executable:
    handle:object
    context_id:int
    owner:object

class CudaStreamBackend:
    def __init__(self,driver,permit,context_id,device_gate_supported,prerequisite_event):
        permit.validate_live()  # before any driver call
        if int(checked(driver.cuCtxGetCurrent()))!=context_id:raise ContractError('not the admitted context')
        if not device_gate_supported:raise ContractError('stream memory operations not verified')
        self.d=driver;self.ctx=context_id;self.permit=permit;self.supported=True
        self.prerequisite=prerequisite_event;self.events={};self.empty=None
        self.streams=[checked(driver.cuStreamCreate(1)) for _ in range(3)]  # NON_BLOCKING
        self.flag=checked(driver.cuMemAlloc(4))

    def context(self,executable):
        if executable.owner is None:raise ContractError('executable owner released')
        return executable.context_id
    def stream_memory_ops_supported(self):return self.supported
    def nonblocking_streams(self):return self.streams

    def reset_gate_and_finish_initialization(self):
        self.permit.validate_live()
        if int(checked(self.d.cuCtxGetCurrent()))!=self.ctx:raise ContractError('context changed')
        ctrl=self.streams[2]
        if self.prerequisite is None:raise ContractError('missing input/preparation completion event')
        checked(self.d.cuStreamWaitEvent(ctrl,self.prerequisite,0))
        checked(self.d.cuMemsetD32Async(self.flag,0,1,ctrl))
        checked(self.d.cuStreamSynchronize(ctrl))  # preparation only, never a phase boundary in real inference
        self.recorded=set()

    def wait_value(self,stream,value):
        # CU_STREAM_WAIT_VALUE_GEQ == 0. No unrecorded-event or SM spin gate.
        checked(self.d.cuStreamWaitValue32(stream,self.flag,value,0))
    def write_value(self,stream,value):checked(self.d.cuStreamWriteValue32(stream,self.flag,value,0))
    def record(self,name,stream):
        if name not in self.events:self.events[name]=checked(self.d.cuEventCreate(0))
        checked(self.d.cuEventRecord(self.events[name],stream));self.recorded.add(name)
    def wait_event(self,stream,name):
        if name not in self.recorded:raise ContractError('unrecorded event is not a dependency')
        checked(self.d.cuStreamWaitEvent(stream,self.events[name],0))
    def launch(self,graph,stream):
        if self.context(graph)!=self.ctx:raise ContractError('cross-context graph')
        checked(self.d.cuGraphLaunch(graph.handle,stream))
    def synchronize_event(self,name):checked(self.d.cuEventSynchronize(self.events[name]))
    def elapsed_ns(self,start,end):
        if start not in self.recorded or end not in self.recorded:raise ContractError('event not recorded in this trial')
        return checked(self.d.cuEventElapsedTime(self.events[start],self.events[end]))*1e6
    def empty_graph(self):
        if self.empty is None:
            graph=checked(self.d.cuGraphCreate(0));execution=checked(self.d.cuGraphInstantiate(graph,0))
            self.empty=(graph,Executable(execution,self.ctx,self))
        return self.empty[1]
    def close_after_join(self):
        checked(self.d.cuEventSynchronize(self.events['joint_end']))
        for event in self.events.values():checked(self.d.cuEventDestroy(event))
        if self.empty:
            checked(self.d.cuGraphExecDestroy(self.empty[1].handle));checked(self.d.cuGraphDestroy(self.empty[0]))
        checked(self.d.cuMemFree(self.flag))
        for stream in self.streams:checked(self.d.cuStreamDestroy(stream))
