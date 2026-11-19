import sqlite3
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from protocol import MODES,STEPS,schedules
from trace import extract

def fixture():
    c=sqlite3.connect(':memory:')
    c.row_factory=sqlite3.Row
    c.executescript("""
        create table TARGET_INFO_GPU(id integer,uuid text);
        create table TARGET_INFO_CUDA_DEVICE(gpuId integer,pid integer,uuid text);
        create table StringIds(id integer,value text);
        create table NVTX_EVENTS(text text,start integer,end integer,globalTid integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,nameId integer,correlationId integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL(graphId integer,graphNodeId integer,globalPid integer,contextId integer,deviceId integer,
            streamId integer,correlationId integer,mangledName integer,shortName integer,gridX integer,gridY integer,gridZ integer,
            blockX integer,blockY integer,blockZ integer,registersPerThread integer,staticSharedMemory integer,dynamicSharedMemory integer,
            start integer,end integer);
        insert into TARGET_INFO_GPU values (1,'uuid');
        insert into TARGET_INFO_CUDA_DEVICE values (1,17,'uuid');
        insert into StringIds values(1,'cudaGraphLaunch_v10000'),(2,'same_main'),(3,'tensor_light');
    """)
    refs={};cs=[]
    for s,mode in zip(STEPS,MODES*4):
        gid=10+MODES.index(mode)
        tags=lambda n:(gid<<32)+n
        source=lambda n:((gid+100)<<32)+n
        refs[mode]=dict(exec_id=gid,targets={k:dict(tools_id=source(i),symbol='same_main',grid=[1,32,1],block=[384,1,1])
                                                 for k,i in [('o',1),('down',2)]},probe_tools_id=source(3))
        c.execute('insert into NVTX_EVENTS values (?,?,?,?)',(f'TARGET_NODE|{s}|{mode}',s*1000,s*1000+90,1))
        c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)',(s*1000+30,s*1000+60,1,1,s))
        def kernel(i,n,start,end):
            probe=i==3
            c.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values ('+','.join(['?']*20)+')',
                      (gid,tags(i),17,1,0,2 if probe else 1,s,n,n,64 if probe else 1,1 if probe else 32,1,
                       32 if probe else 384,1,1,20 if probe else 168,0,1024 if probe else 88064,s*1000+start,s*1000+end))
        kernel(1,2,100,115);kernel(2,2,200,240)
        if mode not in ('original','sham'):
            op,order=mode.split('_')
            start,end=(100,115) if op=='o' else (200,240)
            b={'ab':end,'ba':start-16,'parallel':start}[order]
            kernel(3,3,b,b+16)
        cs.append(dict(iteration=s,mode=mode))
    run=dict(profile='P2',invocation_status='COMPLETED',gpu_uuid='GPU-uuid',run_id='test',config={'precision':'nvfp4'},
             target_nodes={'variants':refs,'trials':[cs]})
    return c,run,dict(status='COMPLETED',known_child_pids=[17])

class TargetTests(unittest.TestCase):
    def make_case(self):
        c,r,g=fixture()
        self.addCleanup(c.close)
        return c,r,g
    def test_distinguishes_spill_and_full_overlap(self):
        c,r,g=self.make_case()
        out=extract(c,r,g)['rows']
        self.assertTrue(all(x['fully_inside'] for x in out if x['mode']=='down_parallel'))
        self.assertEqual(sum(x['mode']=='down_parallel' for x in out),4)
        self.assertTrue(all(x['finish_margin_ns']==-1 for x in out if x['mode']=='o_parallel'))
    def test_wrong_symbol(self):
        c,r,g=self.make_case()
        r['target_nodes']['variants']['down_parallel']['targets']['down']['symbol']='wrong'
        with self.assertRaises(ValueError):extract(c,r,g)
    def test_overlapping_serial_rejected(self):
        c,r,g=self.make_case()
        c.execute('update CUPTI_ACTIVITY_KIND_KERNEL set start=start-8 where graphId=? and shortName=3',(10+MODES.index('o_ab'),))
        with self.assertRaises(ValueError):extract(c,r,g)
    def test_missing_physical_device(self):
        c,r,g=self.make_case()
        r['gpu_uuid']='GPU-another'
        with self.assertRaises(ValueError):extract(c,r,g)
    def test_every_block_balanced(self):
        p=schedules('P0',71011)
        self.assertEqual(len(p),40)
        for i in range(0,len(p),8):
            self.assertEqual({v[0] for v in p[i:i+8]},set(MODES))
            self.assertTrue(all(len(set(v))==1 and len(v)==len(STEPS) for v in p[i:i+8]))
