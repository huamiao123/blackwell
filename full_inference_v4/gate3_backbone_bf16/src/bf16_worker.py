"""One BF16 full-inference control; unchanged original workload and supervision."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve();V4=HERE.parents[2]
sys.path.insert(0,str(V4/'gate3_backbone/src'))
import backbone_worker as base
from bf16_observer import BF16Observer
from common import file_hash

def main():
    base.GraphObserver=BF16Observer
    original_save=base.original.save
    def save(path,data,**kwargs):
        if isinstance(data,dict) and data.get('schema') in ['v4.1','v4.1_failure']:
            data['bf16_control']=dict(worker_sha256=file_hash(HERE),observer_sha256=file_hash(HERE.with_name('bf16_observer.py')),
                note='Matrix collector targets real BF16 WMMA nodes, not NVFP4 nodes')
        original_save(path,data,**kwargs)
    base.original.save=save
    base.main()

if __name__=='__main__':main()
