"""Fake EngineCore contract tests, explicitly not GPU integration results."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import digest
from trace_state import TraceState
from run_full_inference import Hooks


class HookContractTests(unittest.TestCase):
    def test_returned_request_id_and_replay_dispatch_identity(self):
        class Graph:
            def replay(self):return 'executed'
        model=NS(execute_model=lambda scheduler_output,**kwargs:Graph().replay())
        s=NS(num_scheduled_tokens={'actual-id':1},preempted_req_ids=[])
        req=NS(num_computed_tokens=4,num_prompt_tokens=4,prompt_token_ids=[1,2,3,4])
        scheduler=NS(requests={'actual-id':req},schedule=lambda:s)
        core=NS(scheduler=scheduler,model_executor=model)
        client=type('InprocClient',(),{})();client.engine_core=core
        eng=NS(engine_core=client,add_request=lambda request_id,prompt,params:'actual-id',
               step=lambda:[NS(request_id='actual-id',finished=True)],
               output_processor=NS(process_outputs=lambda outputs:None))
        torch=NS(cuda=NS(CUDAGraph=Graph))
        prompt=dict(prompt_id='p',token_ids_sha256=digest([1,2,3,4]),actual_token_length=4)
        hooks=Hooks(NS(llm_engine=eng),[prompt],torch,False)
        hooks.install();hooks.state=TraceState('test',0)
        try:
            eng.add_request('caller-id',{},None)
            self.assertIn('actual-id',hooks.state.requests);self.assertNotIn('caller-id',hooks.state.requests)
            selected=scheduler.schedule();model.execute_model(selected,non_block=True)
            eng.output_processor.process_outputs([NS(request_id='actual-id',new_token_ids=[9])])
            eng.step()
            self.assertEqual(hooks.state.replays[0]['scheduler_iteration'],0)
            self.assertEqual(hooks.state.iterations[0]['graph_replay_ids'],[0])
            self.assertIsNotNone(hooks.state.requests['actual-id']['finished_ns'])
        finally:hooks.restore()


if __name__=='__main__':unittest.main()
