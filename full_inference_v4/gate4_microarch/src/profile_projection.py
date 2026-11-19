"""One counter range around the verified projection's third Graph replay."""
import argparse
from pathlib import Path
import sys

V4 = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(V4/'src'), str(V4/'gate4_spatial/src')]
from common import load
from run_full_inference import require_permit
from validate_qkv_replay import run_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--operator', choices=('qkv_proj','o_proj','gate_up_proj','down_proj'), required=True)
    args = parser.parse_args()
    require_permit(args.config, args.run_id)
    run_dir = V4/'gate4_microarch/raw'/args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    gpu = load(V4/'gate4_spatial/configs/hardware_from_gate3.json')['uuid']
    args.precision = load(args.config)['precision']
    args.run_dir = run_dir
    args.overlap = False
    args.p0_block = None
    run_projection(args, {'gpu_uuid': 'GPU-'+gpu.removeprefix('GPU-')}, profile_node=True)


if __name__=='__main__':
    main()
