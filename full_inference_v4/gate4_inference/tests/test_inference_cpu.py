from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from useful_work import *
from policy import decide
from request_metrics import request_metrics,compare_same_inference
from stage_injection import StageInjection

class InferenceTests(unittest.TestCase):
    def work(self):return UsefulWork('prep','input','layout',['consumer'],True,True,producer_ready=True,required_units=10)
    def test_unused_or_future_dependent_work_is_not_useful(self):
        work=self.work();work.producer_ready=False
        with self.assertRaises(ContractError):work.admission()
        work.producer_ready=True;work.produce('result',10,True)
        with self.assertRaises(ContractError):work.validate_finished()
        work.consume('consumer','result','input',True);self.assertTrue(work.validate_finished())
        with self.assertRaises(ContractError):work.consume('consumer','result','input',True)
    def test_probes_and_old_input_hash_cannot_be_consumer_proof(self):
        work=self.work();work.required_in_original=False
        with self.assertRaises(ContractError):work.admission()
        work=self.work();work.produce('result',10,True)
        with self.assertRaises(ContractError):work.consume('consumer','result','old input',True)
    def test_unmeasured_policy_falls_back(self):
        self.assertFalse(decide(self.work(),{},[],enabled=False).execute_auxiliary)
        self.assertFalse(decide(self.work(),{},[],enabled=True).execute_auxiliary)
    def test_itl_is_per_request_not_batch_progress(self):
        r=request_metrics(dict(request_id='r',arrival_ns=0,completion_ns=50,output_times_ns=[10,20,45],output_token_ids=[1,2,3]))
        self.assertEqual(r['itl_ns'],[10,25]);self.assertEqual(r['tpot_ns'],17.5)
        with self.assertRaises(ContractError):request_metrics(dict(request_id='r',arrival_ns=0,completion_ns=50,output_times_ns=[10,20],output_token_ids=[1,2,3]))
    def test_mixed_is_not_decode_injection(self):
        calls=[];inject=StageInjection('DECODE',lambda *a:calls.append(a))
        row=dict(stage='MIXED',scheduler_iteration=2,scheduled_prefill_tokens=10,scheduled_decode_tokens=8)
        self.assertFalse(inject.on_dispatch(row,dict(inputs_ready=True,dependency_event_recorded=True)))
        self.assertFalse(calls)
    def test_faster_but_unused_output_is_not_inference_gain(self):
        x=dict(work_hash='same',request_ids=['r'],required_output_lengths=[257],auxiliary_required_units=10,consumer_ids=['c'],profile='P0',joint_ns=100)
        with self.assertRaises(ContractError):compare_same_inference(x,dict(x,joint_ns=90,auxiliary_consumed=False,semantic_validation=True))

if __name__=='__main__':unittest.main()
