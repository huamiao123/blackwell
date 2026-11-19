from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from resource_model import *
from partition_model import register_partition_fit
from evidence import classify
from build_projection_inventory import inventory

class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.caps=load(SPATIAL/'configs/hardware_from_gate3.json')
        rows=read_csv(V4/'gate4_residency/analysis/cta_resource_table.csv')
        self.main={r['precision']:{k:(int(v) if k in ['threads','registers_per_thread','static_shared_bytes','dynamic_shared_bytes'] else v)
                    for k,v in r.items()} for r in rows if r['logical_op']=='qkv_proj'}
    def test_global_fit_is_not_partition_fit(self):
        probe=dict(threads=32,registers_per_thread=22,static_shared_bytes=1024,dynamic_shared_bytes=0)
        self.assertEqual(mixed(self.main['nvfp4'],probe,self.caps)['same_sm_theoretical_feasibility'],'NECESSARY_BOUNDS_PASS_NOT_SUFFICIENT')
        gran=load(V4/'gate4_residency/analysis/allocation_granularity_audit.json')
        self.assertEqual(register_partition_fit(self.main['nvfp4'],probe,self.caps,gran)['status'],'REGISTER_PARTITION_MODEL_NO_PLACEMENT')
    def test_heavy_impossible_with_one_nvfp4_cta(self):
        probe=dict(threads=128,registers_per_thread=32,static_shared_bytes=1024,dynamic_shared_bytes=32768)
        r=mixed(self.main['nvfp4'],probe,self.caps)
        self.assertTrue(r['registers_limited'] and r['shared_limited'])
    def test_overlap_or_asserted_booleans_never_direct_proof(self):
        theory=dict(same_sm_theoretical_feasibility='INFEASIBLE_BY_NECESSARY_BOUND')
        r=classify(theory,dict(actual_gpu_intervals_verified=True,overlap_ns=10),dict(both_cta_locations_attested=True))
        self.assertEqual(r['concurrency'],'CONCURRENT_KERNEL_EXECUTION_VERIFIED')
        self.assertEqual(r['same_sm'],'SAME_SM_CO_RESIDENCY_UNVERIFIED')
    def test_main_packing_count_changes_budget(self):
        probe=dict(threads=128,registers_per_thread=32,static_shared_bytes=1024,dynamic_shared_bytes=32768)
        self.assertFalse(mixed(self.main['bf16'],probe,self.caps,n_a=1)['shared_limited'])
        self.assertTrue(mixed(self.main['bf16'],probe,self.caps,n_a=6)['shared_limited'])

    def test_projection_inventory_matches_all_attributed_nodes(self):
        rows = inventory()
        self.assertEqual(len(rows), len({(r['precision'], r['cuda_tools_id']) for r in rows}))
        self.assertEqual(len({(r['precision'], r['logical_op']) for r in rows}), len(
            read_csv(V4 / 'gate4_residency/analysis/cta_resource_table.csv')))
        for row in rows:
            self.assertEqual(row['actual_sm_distribution'], 'UNAVAILABLE')
            self.assertLessEqual(row['grid_distinct_sm_fraction_upper_bound'], 1)

if __name__=='__main__':unittest.main()
