"""Fail-closed identity, memory-lifetime and work-equivalence contracts."""
from dataclasses import dataclass,field
from typing import Any
from g4_common import digest

class ContractError(ValueError):pass

@dataclass(frozen=True)
class Work:
    primary_hash:str
    probe_hash:str
    primary_calls:int=1
    probe_calls:int=1
    cache_protocol_hash:str=''
    def fingerprint(self):return digest(self.__dict__)

@dataclass
class LiveKernel:
    run_id:str
    process_id:int
    context_id:int
    device_uuid:str
    module:str
    tools_id:int
    kernel_parameters:Any
    graph_owner:Any
    model_owner:Any
    input_ready:bool=False
    inputs_immutable_until_join:bool=False
    producer_dependencies_complete:bool=False
    output_region_verified:bool=False
    reference_output_verified:bool=False
    same_kernel_signature_verified:bool=False
    cache_protocol_verified:bool=False
    mutable_state_reset_verified:bool=False
    non_kernel_dependencies_supported:bool=False
    closed:bool=False

    def validate(self,current_process,current_context,current_device):
        if self.closed:raise ContractError('kernel lease closed')
        if (self.process_id,self.context_id,self.device_uuid)!=(current_process,current_context,current_device):
            raise ContractError('stale process/context/device identity')
        if self.graph_owner is None or self.model_owner is None or self.kernel_parameters is None:
            raise ContractError('graph/model/argument lifetime not pinned')
        for name in ['input_ready','inputs_immutable_until_join','producer_dependencies_complete',
                     'output_region_verified','reference_output_verified','same_kernel_signature_verified',
                     'cache_protocol_verified','mutable_state_reset_verified','non_kernel_dependencies_supported']:
            if not getattr(self,name):raise ContractError('SINGLE_NODE_REPLAY_NOT_READY: '+name)
        return True

def validate_work_equal(records):
    fingerprints={r['work_hash'] for r in records}
    if len(fingerprints)!=1:raise ContractError('unequal logical work/cache protocol')
    if any(r.get('work_completed') is not True for r in records):raise ContractError('partial work is not completed work')
    if any(r.get('correctness_passed') is not True for r in records):raise ContractError('unchecked output')

def require_scope(record,scope):
    if record.get('measurement_scope')!=scope:raise ContractError('scope mismatch; stage/full-Graph data cannot answer single-kernel H1')
