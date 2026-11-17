from pathlib import Path
from types import SimpleNamespace as NS
import sys,unittest,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contracts import LiveKernel,ContractError,require_scope
from replay import DiagnosticNode,ancestor_closure
from stream_protocol import submit
from plan import MODES,probe_specs,GRIDS,make_plan
from probe_reference import work_accounting,fma_oracle,tensor_oracle,tensor_a,tensor_b
from metrics import paired,overlap,confidence,completed_within_primary
from compile_probes import parse_resources,parse_sass

class Recorder:
    def __init__(self):self.calls=[];self.events=set()
    def context(self,x):return x.context
    def stream_memory_ops_supported(self):return True
    def reset_gate_and_finish_initialization(self):self.calls.append(('reset',))
    def nonblocking_streams(self):return 'sa','sb','ctrl'
    def wait_value(self,s,v):self.calls.append(('wait_value',s,v))
    def write_value(self,s,v):self.calls.append(('release',s,v))
    def record(self,e,s):self.events.add(e);self.calls.append(('record',e,s))
    def wait_event(self,s,e):
        if e not in self.events:raise AssertionError('unrecorded event wait')
        self.calls.append(('dependency',s,e))
    def launch(self,g,s):self.calls.append(('launch',g.role,s))
    def synchronize_event(self,e):self.calls.append(('join',e))
    def empty_graph(self):return NS(role='EMPTY',context=1)

class ProtocolTests(unittest.TestCase):
    def test_all_nine_controls_and_gate_order(self):
        for mode in MODES:
            r=Recorder();submit(r,mode,NS(role='A',context=1),NS(role='B',context=1))
            release=next(i for i,c in enumerate(r.calls) if c[0]=='release')
            launches=[(i,c) for i,c in enumerate(r.calls) if c[0]=='launch']
            self.assertTrue(all(i<release for i,c in launches))
            if mode=='parallel_ba':self.assertEqual([c[1] for i,c in launches],['B','A'])
            if mode=='same_stream_control':self.assertEqual({c[2] for i,c in launches},{'sa'})
            if mode.startswith('serial') or mode=='non_overlap_control':
                self.assertTrue(any(c[0]=='dependency' and c[2].endswith('_end') for c in r.calls))
    def test_context_and_scope_cannot_be_faked(self):
        with self.assertRaises(ContractError):submit(Recorder(),'parallel_ab',NS(context=1),NS(context=2))
        with self.assertRaises(ContractError):require_scope(dict(measurement_scope='FULL_GRAPH'),'SINGLE_NODE')

class WorkTests(unittest.TestCase):
    def test_grid_scans_keep_actual_work_constant(self):
        for spec in probe_specs():
            rows=[work_accounting(spec,g) for g in GRIDS]
            self.assertEqual(len({r['work_hash'] for r in rows}),1)
            self.assertEqual(len({r['flops'] for r in rows}),1)
            self.assertTrue(all(r['nonempty_ctas']==g for r,g in zip(rows,GRIDS)))
    def test_oracles_are_bounded_and_staging_invariant(self):
        self.assertTrue(all(.25<x<1 for x in fma_oracle(512)))
        reference=tensor_oracle(3)
        for i in range(16):
            for j in range(16):
                for chunk in range(8):
                    self.assertEqual(reference[i][j]/3,sum(tensor_a(i,16*chunk+k)*tensor_b(16*chunk+k,j) for k in range(16)))
    def test_compiler_formats_and_wrong_dtype(self):
        data='Function fp32_light:\n REG:22 STACK:0 SHARED:1024\n Function tensor_light:\n REG:43 SHARED:1024\n'
        self.assertEqual(parse_resources(data)['fp32_light']['registers_per_thread'],22)
        data=' Function : tensor_light\n HMMA.16816.F32.BF16 R1,R2;\n Function : fp32_light\n FFMA R1,R2,R3,R4;\n'
        self.assertEqual(parse_sass(data)['tensor_light']['bf16_mma_static_instructions'],1)
        self.assertEqual(parse_sass(data.replace('.BF16','.F16'))['tensor_light']['bf16_mma_static_instructions'],0)

class MetricsTests(unittest.TestCase):
    def block(self):
        base=dict(work_hash='same',work_completed=True,correctness_passed=True,profile='P0',measurement_scope='EVENT_ENVELOPE')
        return dict(a_alone=dict(base,a_ns=100,a_completed_units=1,b_completed_units=0),
          b_alone=dict(base,b_ns=80,a_completed_units=0,b_completed_units=16),
          serial_ab=dict(base,joint_ns=195,serial_measured=True,a_completed_units=1,b_completed_units=16),
          parallel_ab=dict(base,a_ns=104,b_ns=90,joint_ns=120,a_completed_units=1,b_completed_units=16))
    def test_measured_serial_and_two_direction_effects(self):
        r=paired(self.block(),'ab');self.assertAlmostEqual(r['primary_interference'],.04)
        self.assertAlmostEqual(r['probe_throughput_retention'],80/90);self.assertEqual(r['joint_speedup'],195/120)
    def test_partial_work_profile_and_arithmetic_serial_rejected(self):
        for key,value in [('b_completed_units',8),('profile','P2'),('work_hash','different')]:
            b=self.block();b['parallel_ab'][key]=value
            with self.assertRaises(ContractError):paired(b,'ab')
        b=self.block();b['serial_ab']['serial_measured']=False
        with self.assertRaises(ContractError):paired(b,'ab')
    def test_overlap_not_tail_proration_or_independent_steps(self):
        self.assertEqual(overlap((100,180),(140,200)),(40,40/60))
        self.assertEqual(completed_within_primary((0,10),(2,8),100),100)
        self.assertIsInstance(completed_within_primary((0,10),(2,20),100),str)
        with self.assertRaises(ContractError):confidence([1,2,3,4])

class ReplayTests(unittest.TestCase):
    def lease(self,ready=True):
        l=LiveKernel('live',12,9,'uuid','model.layers.18.self_attn.qkv_proj',5,object(),object(),object())
        for k in ['input_ready','inputs_immutable_until_join','producer_dependencies_complete','output_region_verified',
                  'reference_output_verified','same_kernel_signature_verified','cache_protocol_verified',
                  'mutable_state_reset_verified','non_kernel_dependencies_supported']:setattr(l,k,ready)
        return l
    def test_old_addresses_or_missing_input_proof_refused(self):
        with self.assertRaises(ContractError):self.lease().validate(13,9,'uuid')
        with self.assertRaises(ContractError):self.lease(False).validate(12,9,'uuid')
    def test_clone_only_and_cannot_free_before_join(self):
        calls=[]
        def f(name,value):
            def call(*args):calls.append((name,args));return (0,value)
            return call
        driver=NS(cuGraphKernelNodeGetParams=f('get','params'),cuGraphCreate=f('create','diagnostic'),
            cuGraphAddKernelNode=f('add','newnode'),cuGraphKernelNodeCopyAttributes=f('copy',None),
            cuGraphDestroy=f('destroy',None),cuGraphInstantiate=f('instantiate','exec'),
            cuGraphLaunch=f('launch',None),cuGraphExecDestroy=f('destroy_exec',None))
        lease=self.lease(False);candidate=DiagnosticNode.copy_during_live_capture(driver,'ORIGINAL',lease,(12,9,'uuid'))
        with self.assertRaises(ContractError):candidate.instantiate()
        for key in list(lease.__dict__):
            if isinstance(getattr(lease,key),bool) and key!='closed':setattr(lease,key,True)
        candidate.instantiate();candidate.launch('owned_stream')
        with self.assertRaises(ContractError):candidate.close_after_join(False)
        candidate.close_after_join(True)
        self.assertFalse(any(name.startswith('set') for name,args in calls))
        self.assertEqual(next(args for name,args in calls if name=='copy'),('newnode','ORIGINAL'))
    def test_prefix_plan_flags_external_events(self):
        nodes=[dict(index=0,node_type='event_wait',dependencies=[]),dict(index=1,node_type='kernel',dependencies=[0])]
        self.assertTrue(ancestor_closure(nodes,1)['requires_external_dependency_review'])

if __name__=='__main__':unittest.main()
