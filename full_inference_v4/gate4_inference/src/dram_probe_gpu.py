"""Standalone SM120 streaming acceptance and shared fixed-work read probe."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

SRC = Path(__file__).resolve()
V4 = SRC.parents[2]
sys.path[:0] = [str(V4/'src'), str(V4/'gate4_spatial/src')]
from common import file_hash
from dram_reference import TILE_WORDS, streaming_work
from replay import checked

GRID = 128
SEED = 94137
SYMBOL = 'dram_read_light'
COMPILE = V4/'gate4_spatial/build/dram_20261119_031204_utc'


class StreamProbe:
    def __init__(self, torch):
        import numpy as np
        from cuda.bindings import driver
        self.driver = driver
        self.torch = torch
        torch.cuda.set_device(0)
        self.context_anchor = torch.empty(1, dtype=torch.int32, device='cuda')
        self.device = torch.cuda.get_device_properties(0)
        self.l2 = checked(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, checked(driver.cuCtxGetDevice())))
        self.binary = COMPILE/'dram_streaming_sm120.cubin'
        self.sha = file_hash(self.binary)
        if self.sha != json.loads((COMPILE/'compile_manifest.json').read_bytes())['cubin_sha256']:
            raise ValueError('DRAM cubin hash differs from archived build')
        self.module = checked(driver.cuModuleLoad(os.fsencode(self.binary)))
        words = streaming_work('read',self.l2,GRID,'light')['region_bytes']//4
        self.host = np.random.default_rng(SEED).integers(0,2**32,size=words,dtype=np.uint32)
        self.source = torch.from_numpy(self.host.view(np.int32)).to('cuda')
        self.digest = np.empty(words//TILE_WORDS,dtype=np.uint32)
        for tile in range(0,len(self.digest),128):
            first = tile*TILE_WORDS
            end = min(first+128*TILE_WORDS,words)
            weights = np.arange(first,end,dtype=np.uint32) | np.uint32(1)
            self.digest[tile:end//TILE_WORDS] = (self.host[first:end]*weights).reshape(-1,TILE_WORDS).sum(axis=1,dtype=np.uint32)
        self.stream = torch.cuda.Stream()

    def select(self,symbol):
        _,op,variant = symbol.split('_')
        self.symbol = symbol
        self.work = streaming_work(op,self.l2,GRID,variant)
        self.fn = checked(self.driver.cuModuleGetFunction(self.module,symbol.encode()))
        enum = self.driver.CUfunction_attribute
        attrs = {k:checked(self.driver.cuFuncGetAttribute(getattr(enum,'CU_FUNC_ATTRIBUTE_'+k),self.fn))
                 for k in ('NUM_REGS','SHARED_SIZE_BYTES','LOCAL_SIZE_BYTES','MAX_THREADS_PER_BLOCK')}
        self.spec = dict(symbol=symbol,**self.work,**attrs)
        return self.spec

    def launch(self, output, seed=SEED):
        d = self.driver
        args = ((self.source.data_ptr(),output.data_ptr(),self.work['tiles'],seed),
                (ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32))
        checked(d.cuLaunchKernel(self.fn,GRID,1,1,self.work['threads_per_cta'],1,1,self.work['dynamic_shared_bytes'],
                                 d.CUstream(init_value=self.stream.cuda_stream),args,0))

    def validate(self, output, seed=SEED):
        import numpy as np
        actual = output.cpu().numpy().view(np.uint32).reshape(-1)
        op = self.work['operation']
        if op == 'write':
            for first in range(0,len(actual),2**20):
                end = min(first+2**20,len(actual))
                words = (np.arange(first,end,dtype=np.uint32)^np.uint32(seed))*np.uint32(0x9e3779b9)
                if not np.array_equal(actual[first:end],words^(words>>16)):
                    raise ValueError('wrong streaming write')
        elif not np.array_equal(actual,self.host if op=='copy' else self.digest):
            raise ValueError('wrong stream output: '+op)
        return dict(correctness_passed=True, checked_uint32=len(actual),
                    output_sha256=hashlib.sha256(memoryview(actual)).hexdigest())

    def alone(self, output, repeats=6):
        samples = []
        for _ in range(repeats):
            begin,end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record(self.stream)
            self.launch(output)
            end.record(self.stream)
            end.synchronize()
            samples.append(begin.elapsed_time(end)*1000)
        return dict(gpu_us=samples[1:],median_us=statistics.median(samples[1:]),**self.validate(output))


def main():
    parser=argparse.ArgumentParser()
    for key in ('config','profile','output','run-id','condition'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--execute',action='store_true',required=True)
    a=parser.parse_args()
    from run_full_inference import require_permit
    require_permit(a.config,a.run_id)
    if a.profile not in ('P0','P1') or a.condition!='acceptance':
        raise ValueError('unsupported acceptance case')
    import torch
    torch.cuda.init()
    p=StreamProbe(torch)
    short = torch.empty(len(p.digest),dtype=torch.int32,device='cuda')
    full = torch.empty_like(p.source)
    profiled = a.profile=='P1'
    results=[]
    if profiled:
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range('DRAM_IDLE_CONTROL'):
            torch.cuda.synchronize()
            time.sleep(.035)
    for op in ('read','write','copy'):
        for variant in ('light','heavy'):
            spec=p.select('dram_'+op+'_'+variant)
            out=short if op=='read' else full
            out.fill_(-1737653)
            torch.cuda.synchronize()
            launches = 41 if profiled and op=='write' else 9
            with torch.cuda.nvtx.range('DRAM_PASS|'+p.symbol):
                values=p.alone(out,launches)
            results.append(dict(**spec,**values,launches=launches))
    if profiled:
        torch.cuda.profiler.stop()
    data=dict(run_id=a.run_id, profile=a.profile, gpu_uuid=str(p.device.uuid), l2_bytes=p.l2,grid=GRID,
              cubin_sha256=p.sha, source_sha256=file_hash(SRC),seed=SEED,results=results,
              status='STANDALONE_FULL_OUTPUTS_VERIFIED')
    with open(a.output,'x',encoding='utf-8') as f:
        json.dump(data,f,indent=2)
    print(data['status'], [(r['symbol'],round(r['median_us'])) for r in results],flush=True)


if __name__=='__main__':
    main()
