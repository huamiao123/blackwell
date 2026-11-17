import copy
import datetime as dt
import json
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'local_cache').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from common import classify_schedule,duration,intersection,digest,validate_config,load,paired_metrics
from trace_state import TraceState
from run_supervised import validate_window,controllers,parse_gpu_state
from run_full_inference import prepare,require_permit
from analyze_timeline import analyze
from map_kernels import map_rows
from analyze_operators import operator_timings
from analyze_resources import summarize_samples
from analyze_overlap import physical_overlap
from analyze_runs import compare_overhead
from run_teacher_forcing import make_plan
from validate_probe import validate_artifact


class MeasurementTests(unittest.TestCase):
    def test_interval_oracle(self):
        rng=random.Random(410)
        for _ in range(200):
            a=[sorted([rng.randrange(30),rng.randrange(30)]) for _ in range(12)]
            b=[sorted([rng.randrange(30),rng.randrange(30)]) for _ in range(12)]
            aa={x for s,e in a for x in range(s,e)};bb={x for s,e in b for x in range(s,e)}
            self.assertEqual(duration(a),len(aa));self.assertEqual(duration(intersection(a,b)),len(aa&bb))
        self.assertEqual(duration([(0,10),(5,15)]),15)

    def test_mixed_boundary_from_pre_schedule_snapshot(self):
        r=classify_schedule({'a':dict(computed=2000,prompt_length=2048),'b':dict(computed=2200,prompt_length=2048)}, {'a':48,'b':1})
        self.assertEqual((r['stage'],r['scheduled_prefill_tokens'],r['scheduled_decode_tokens']),('MIXED',48,1))
        self.assertEqual(classify_schedule({}, {})['stage'],'EMPTY')
        r=classify_schedule({'a':dict(computed=3,prompt_length=4)},{'a':2})
        self.assertEqual((r['scheduled_prefill_tokens'],r['scheduled_decode_tokens']),(1,1))

    def test_per_request_completion_is_not_batch_completion(self):
        st=TraceState('test',0)
        p=dict(prompt_id='p',token_ids_sha256='h',actual_token_length=4)
        st.enqueue('a',p,0);st.enqueue('b',p,100)
        st.output('a',[1],1000000);st.output('a',[2],2000000);st.finish('a',2100000)
        st.output('b',[1],3000000);st.output('b',[2],6000000);st.finish('b',6200000)
        a,b=st.summaries()
        self.assertEqual(a['model_request_ms'],2.1);self.assertAlmostEqual(b['tpot_ms'],3)
        self.assertNotEqual(a['model_request_ms'],b['model_request_ms'])

    def test_bundled_outputs_never_called_token_itl(self):
        st=TraceState('x',0);st.enqueue('a',dict(prompt_id='p',token_ids_sha256='h',actual_token_length=4),0)
        st.output('a',[1,2],100);st.output('a',[3],200);st.finish('a',300)
        self.assertIn('not token ITL',st.summaries()[0]['itl_scope'])

    def test_first_output_not_decode_step(self):
        st=TraceState('x',0);st.enqueue('a',dict(prompt_id='p',token_ids_sha256='h',actual_token_length=4),0)
        st.schedule({'a':dict(computed=0,prompt_length=4)},{'a':4},0,1)
        st.output('a',[8],10)
        for i in range(2):
            st.schedule({'a':dict(computed=4+i,prompt_length=4)},{'a':1},i+1,i+2)
            st.output('a',[9+i],20+i)
        st.finish('a',30);self.assertTrue(st.verify_fixed(2))
        with self.assertRaises(ValueError):st.verify_fixed(3)

    def test_config_context_and_matching(self):
        c,ps=prepare(ROOT/'configs/E2_bf16.json');self.assertEqual(len(ps),8)
        c['engine']['max_model_len']=2048
        with self.assertRaises(ValueError):validate_config(c,ps)
        c,ps=prepare(ROOT/'configs/E7_bf16.json');self.assertEqual(max(p['actual_token_length'] for p in ps),4096)
        c['engine']['max_model_len']=4096
        with self.assertRaises(ValueError):validate_config(c,ps)
        for e in ['E1','E2','E3','E4','E5','E6','E7']:
            a=load(ROOT/f'configs/{e}_bf16.json');b=load(ROOT/f'configs/{e}_nvfp4.json')
            self.assertEqual(a['prompt_work_hash'],b['prompt_work_hash']);self.assertEqual(a['engine'],b['engine'])

    def test_refuse_direct_gpu_before_imports(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'GPU_DISABLED'):require_permit(ROOT/'configs/E1_bf16.json','x')

    def test_window_disabled_expired_wrong_scope(self):
        c,_=prepare(ROOT/'configs/E1_bf16.json')
        w=dict(enabled=True,gate=2,reviewed_gates=[0,1],hostname='host',allowed_configs=['E1_bf16'],
               not_before='2026-01-01T00:00:00+00:00',expires_at='2026-01-01T01:00:00+00:00',authorization_reference='unit fixture')
        now=dt.datetime(2026,1,1,0,30,tzinfo=dt.timezone.utc).timestamp()
        self.assertGreater(validate_window(w,c,now,'host'),now)
        for key,value in [('enabled',False),('gate',6),('reviewed_gates',[]),('hostname','other'),('allowed_configs',[])]:
            ww=dict(w);ww[key]=value
            with self.assertRaises(ValueError):validate_window(ww,c,now,'host')
        with self.assertRaises(ValueError):validate_window(w,c,now+7200,'host')

    def test_foreign_process_and_controller_parsers(self):
        text='\n    Process ID : 999\n        Type : C\n        Process Name : other\n'
        self.assertEqual(parse_gpu_state(text),[999])
        self.assertTrue(controllers('12 1 1000 python python /other/wait_and_launch_live_controller_v79.py'))
        self.assertFalse(controllers('12 1 1001 python python run_supervised.py'))

    def test_same_work_metrics_use_measured_serial_and_b_tail(self):
        a=dict(a_ms=100,a_work_hash='a');b=dict(b_span_ms=30,b_work_hash='b')
        sab=dict(a_work_hash='a',b_work_hash='b',joint_ms=150)
        sba=dict(sab,joint_ms=140)
        p=dict(a_work_hash='a',b_work_hash='b',a_ms=102,b_span_ms=40,joint_ms=120,a_end_ns=102,b_end_ns=120)
        r=paired_metrics(a,b,sab,sba,p)
        self.assertAlmostEqual(r['ia'],.02);self.assertEqual(r['rb'],.75)
        self.assertAlmostEqual(r['s_fastest_serial'],140/120);self.assertFalse(r['b_complete_before_a_end'])
        p['b_work_hash']='different'
        with self.assertRaises(ValueError):paired_metrics(a,b,sab,sba,p)

    def test_partial_work_not_prorated(self):
        r=physical_overlap([(0,10)],[(2,4),(8,12)],10)
        self.assertEqual(r['fully_completed_b_kernels'],1);self.assertEqual(r['partial_b_kernels'],1)
        self.assertEqual(r['kernel_intersection_ns'],4)

    def test_resource_time_weights_and_gaps(self):
        base=dict(typeId=0,metricId=0,name='DRAM Read Bandwidth [Throughput %]')
        ss=[dict(base,timestamp=t,value=v) for t,v in [(0,10),(100,30),(200,99)]]
        r=summarize_samples(ss,{'all':[(0,300)]})[0]
        self.assertEqual(r['time_weighted_mean'],20);self.assertEqual(r['interval_coverage'],2/3)
        ss=[dict(base,timestamp=t,value=10) for t in [0,100,200,900,1000]]
        r=summarize_samples(ss,{'all':[(0,1000)]})[0]
        self.assertEqual(r['valid_interval_ns'],300)

    def test_clock_exports_not_repaired(self):
        s=[dict(typeId=0,metricId=2,name='GPC Clock Frequency',timestamp=t,value=v) for t,v in [(0,-10),(100,9e9)]]
        self.assertEqual(summarize_samples(s,{'full':[(0,200)]})[0]['time_weighted_mean'],'UNAVAILABLE')

    def test_operator_unknown_and_fusion_union(self):
        r=dict(kernel_name='same',contextId=1,graphId=2,graphNodeId=3,activity_kind='KERNEL',stage='DECODE',start=0,end=10)
        m=map_rows([r],None,'sha')[0];self.assertEqual(m['logical_op'],'UNKNOWN')
        contract=dict(trace_sha256='wrong',entries=[])
        with self.assertRaises(ValueError):map_rows([r],contract,'sha')
        for x in [m]:x.update(logical_op='fused_gate_up',layer_id=0,mapping_status='VERIFIED')
        row2=dict(m,start=5,end=15)
        self.assertEqual(operator_timings([m,row2])[0]['kernel_union_ns'],15)

    def test_profile_pair_must_match_config_and_code(self):
        a=dict(config_file_sha256='a',prompt_manifest_hash='b',code_sha256={'x':'1'},runs=[dict(full_model_wall_ms=100)])
        b=dict(a,profile='P1',runs=[dict(full_model_wall_ms=110)])
        self.assertAlmostEqual(compare_overhead(a,b)['overhead_ratio'],.1)
        b['code_sha256']={'x':'2'}
        with self.assertRaises(ValueError):compare_overhead(a,b)

    def test_teacher_forcing_plan_not_full_prefill_substitute(self):
        p=dict(token_ids_sha256='p')
        plan=make_plan([p],{'p':[1]*257},[1,2])
        self.assertEqual(len(plan['plans'][0]['decode_inputs']),256)
        self.assertIn('DISABLED',plan['status'])
        with self.assertRaises(ValueError):make_plan([p],{'p':[1]*256},[1])

    def test_probe_admission_requires_gpu_checks(self):
        with self.assertRaises(ValueError):validate_artifact(dict(kind='fp32'))

    def test_correlations_not_cpu_kernel_time_overlap(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'local_cache') as tmp:
            path=Path(tmp)/'fixture.sqlite';c=sqlite3.connect(path)
            c.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,correlationId INTEGER,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,correlationId INTEGER,globalPid INTEGER,contextId INTEGER,deviceId INTEGER,streamId INTEGER,graphId INTEGER,graphNodeId INTEGER);
            CREATE TABLE TARGET_INFO_GPU(uuid TEXT,busLocation TEXT);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(uuid TEXT);
            INSERT INTO StringIds VALUES(1,'native_e2m1_kernel'),(2,'cudaGraphLaunch');
            INSERT INTO NVTX_EVENTS VALUES(0,200,'FULL_INFERENCE',4),(5,15,'ITERATION_DISPATCH|0|DECODE',4);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(10,11,2,42,4);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,130,1,42,99,1,0,7,2,3),(140,150,1,999,99,1,0,7,2,4);
            INSERT INTO TARGET_INFO_GPU VALUES('same','0000:27:00.0');
            INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES('same');''')
            c.commit();c.close()
            summary,rows=analyze(path,Path(tmp)/'analysis')
            self.assertEqual(rows[0]['stage'],'DECODE');self.assertEqual(rows[1]['stage'],'UNKNOWN')
            self.assertEqual(summary['kernel_time_union_ns'],40)
            self.assertEqual(summary['gpu_identity_status'],'VERIFIED_METADATA')


if __name__=='__main__':
    (ROOT/'local_cache').mkdir(exist_ok=True)
    unittest.main()
