"""Probe admission contract and pressure plan, without importing GPU libraries."""
import argparse
from common import load


def pressure_plan():
    return dict(status='DESIGN_ONLY',fp32=dict(logical_rows=8192,block_threads=128,fma_chains=4,
                    iterations_per_row=8000,grid_ctas=[4,16,64],distribution='grid-stride over fixed logical rows'),
                dram=dict(input_bytes=268435456,output_bytes=268435456,passes=128,grid_ctas=[64,256,1024],
                          distribution='same full-buffer bytes for every level'),
                tensor=dict(logical_gemm_tiles=64,grid_ctas=[4,16,64],
                            distribution='persistent grid-stride over fixed tile tasks; dimensions frozen before Gate 6'),
                caution='Changing resource_overlap_v1 FP32 ctas changes work; it cannot directly implement this fixed-work pressure scan.')


def validate_artifact(d):
    if not d.get('gpu_correctness_passed'):raise ValueError('GPU correctness not proven')
    if not d.get('output_validation_passed'):raise ValueError('no checked result')
    if not d.get('compiler_instruction_verified'):raise ValueError('instruction path unverified')
    if d['kind']=='dram' and (d['input_bytes']<=d['l2_capacity_bytes'] or not d.get('streaming_access_verified')):
        raise ValueError('not a verified DRAM probe')
    if not d.get('standalone_runs') or len(d['standalone_runs'])<5:raise ValueError('insufficient standalone repeats')
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact');a=p.parse_args()
    print(validate_artifact(load(a.artifact)) if a.artifact else pressure_plan())
