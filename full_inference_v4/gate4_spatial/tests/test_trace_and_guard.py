from pathlib import Path
import sys,sqlite3,tempfile,unittest,subprocess
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze_trace import analyze
from contracts import ContractError
from fallback_stage import StageFallback

class TraceTests(unittest.TestCase):
    def fixture(self,path,same_stream=False):
        c=sqlite3.connect(path)
        c.executescript('''
        create table TARGET_INFO_GPU(uuid text);
        create table TARGET_INFO_CUDA_DEVICE(gpuId int,cudaId int,pid int,uuid text);
        create table StringIds(id int,value text);
        create table NVTX_EVENTS(start int,end int,globalTid int,text text,textId int);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(correlationId int,start int,end int,globalTid int);
        create table CUPTI_ACTIVITY_KIND_KERNEL(correlationId int,start int,end int,globalPid int,contextId int,deviceId int,streamId int,demangledName int);
        insert into TARGET_INFO_GPU values('uuid');
        insert into TARGET_INFO_CUDA_DEVICE values(0,0,1,'uuid');
        insert into StringIds values(1,'primary_kernel'),(2,'probe_kernel');
        insert into NVTX_EVENTS values(0,20,4,'G4|case|0|parallel_ab|A',null),(20,40,4,'G4|case|0|parallel_ab|B',null);
        insert into CUPTI_ACTIVITY_KIND_RUNTIME values(1,1,2,4),(2,21,22,4);
        insert into CUPTI_ACTIVITY_KIND_KERNEL values(1,100,180,1,7,0,11,1),(2,140,200,1,7,0,12,2);
        ''')
        if same_stream:c.execute('update CUPTI_ACTIVITY_KIND_KERNEL set streamId=11')
        for name in ['gridX','gridY','gridZ','blockX','blockY','blockZ','registersPerThread','staticSharedMemory','dynamicSharedMemory']:
            c.execute('alter table CUPTI_ACTIVITY_KIND_KERNEL add column '+name+' int')
        c.execute('update CUPTI_ACTIVITY_KIND_KERNEL set gridX=48,gridY=1,gridZ=1,blockX=384,blockY=1,blockZ=1,registersPerThread=168,staticSharedMemory=0,dynamicSharedMemory=88064 where correlationId=1')
        c.commit();c.close()
    def test_gpu_times_outside_cpu_range_still_join_by_correlation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.sqlite';self.fixture(p)
            rows,unmapped=analyze(p,'uuid');self.assertFalse(unmapped)
            self.assertEqual(rows[0]['overlap_ns'],40)
            self.assertEqual(rows[0]['a_start_ns'],100)
            self.assertEqual(rows[0]['a_grid'],[48,1,1])
            self.assertEqual(rows[0]['a_dynamic_shared_bytes'],88064)
            self.assertEqual(rows[0]['b_grid'],'UNAVAILABLE')
            self.assertEqual(rows[0]['b_registers_per_thread'],'UNAVAILABLE')
            self.assertEqual(rows[0]['same_sm'],'SAME_SM_CO_RESIDENCY_UNVERIFIED')
    def test_parallel_same_stream_and_wrong_gpu_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.sqlite';self.fixture(p,True)
            with self.assertRaises(ContractError):analyze(p,'uuid')
            with self.assertRaises(ContractError):analyze(p,'wrong')
    def test_cpu_review_execute_never_imports_gpu_runtime(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.json';p.write_text(json.dumps(dict(review_status='CPU_REVIEW_PENDING',gpu_authorized=False)))
            r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'src/review_gate.py'),'--config',str(p),'--execute'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('GPU_DISABLED_CPU_PREPARATION',r.stderr)
    def test_fallback_does_not_inject_into_empty_or_mixed_as_decode(self):
        calls=[];scheduler=object()
        ex=NS(execute_model=lambda *a,**k:calls.append('original'))
        hooks=NS(core=NS(model_executor=ex),state=object(),schedule_ids={id(scheduler):dict(stage='EMPTY',active_request_count=0)})
        f=StageFallback(hooks,'DECODE',lambda row:None,lambda *a:calls.append('probe'));f.install()
        ex.execute_model(scheduler);f.restore();self.assertEqual(calls,['original'])

if __name__=='__main__':unittest.main()
