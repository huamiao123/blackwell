"""Real-consumer and dependency contracts; synthetic probes never qualify."""
from dataclasses import dataclass,field
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gate4_spatial/src'))
from contracts import ContractError
from g4_common import digest

@dataclass
class UsefulWork:
    task_id:str
    input_hash:str
    output_contract_hash:str
    consumer_ids:list[str]
    required_in_original:bool
    semantics_unchanged:bool
    producer_ready:bool=False
    correctness_passed:bool=False
    produced_hash:str|None=None
    consumed_by:set[str]=field(default_factory=set)
    completed_units:int=0
    required_units:int=0
    consumed_after_ready:bool=False

    def admission(self):
        if not self.required_in_original:raise ContractError('work added only to occupy resources')
        if not self.consumer_ids:raise ContractError('no real downstream consumer')
        if len(set(self.consumer_ids))!=len(self.consumer_ids):raise ContractError('duplicate consumer declaration')
        if not self.input_hash or not self.output_contract_hash:raise ContractError('missing data identity')
        if not self.producer_ready:raise ContractError('upstream result not ready')
        if not self.semantics_unchanged:raise ContractError('numerical semantics changed')
        if self.produced_hash is not None:raise ContractError('duplicate preparation')
        if self.required_units<=0:raise ContractError('missing effective work definition')

    def produce(self,output_hash,units,correct):
        self.admission()
        if units!=self.required_units or not correct:raise ContractError('incomplete/incorrect output')
        self.produced_hash=output_hash;self.completed_units=units;self.correctness_passed=correct

    def consume(self,consumer,output_hash,input_hash,after_ready):
        if not self.correctness_passed or self.produced_hash!=output_hash or self.input_hash!=input_hash:
            raise ContractError('stale or unvalidated auxiliary result')
        if consumer not in self.consumer_ids or consumer in self.consumed_by or not after_ready:
            raise ContractError('unexpected/duplicate/early consumer')
        self.consumed_by.add(consumer);self.consumed_after_ready=after_ready

    def validate_finished(self):
        if not self.required_in_original or not self.semantics_unchanged or not self.consumer_ids:
            raise ContractError('no valid required useful-work contract')
        if set(self.consumer_ids)!=self.consumed_by:raise ContractError('auxiliary output not actually consumed')
        if self.completed_units!=self.required_units or not self.correctness_passed:raise ContractError('invalid finished work')
        return True

def inference_work_hash(requests,auxiliary):
    return digest(dict(requests=requests,auxiliary=auxiliary))
