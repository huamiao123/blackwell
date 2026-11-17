"""CPU oracles and exact logical-work accounting. No CUDA imports."""
import math,struct
from g4_common import digest

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def fp32_seed(index):return .25+(index%64)/128

def fma_oracle(rounds):
    if not 0<rounds<=4096:raise ValueError('oracle validated domain rounds=1..4096')
    result=[]
    for i in range(64):
        x=fp32_seed(i)
        for _ in range(rounds):
            # Chosen bounded binary operands have an exact product+sum within
            # binary64 precision, followed by the one required FP32 rounding.
            x=f32(x*float.fromhex('0x1.000002p0')+float.fromhex('0x1p-20'))
        result.append(x)
    return result

def tensor_a(i,k):return .5+((i+k)%4)*.125
def tensor_b(k,j):return .25+((k+j)%4)*.0625
def tensor_oracle(rounds):
    if not 0<rounds<=4096:raise ValueError('oracle domain')
    return [[rounds*sum(tensor_a(i,k)*tensor_b(k,j) for k in range(16)) for j in range(16)] for i in range(16)]

def work_accounting(spec,grid):
    if grid<=0:raise ValueError('grid')
    work_hash=digest({k:v for k,v in spec.items() if k!='logical_work_hash'})
    if spec['probe']=='fp32_fma':
        groups=math.ceil(spec['items']/512)
        flops=spec['items']*16*spec['rounds']*2
        return dict(fma_operations=flops//2,flops=flops,input_bytes=spec['items']*16*4,
                    output_bytes=spec['items']*16*4,virtual_groups=groups,
                    nonempty_ctas=min(grid,groups),work_hash=work_hash)
    groups=math.ceil(spec['tiles']/(spec['threads']//32))
    return dict(mma_operations=spec['tiles']*spec['mma_rounds'],flops=spec['expected_flops'],
                input_bytes=groups*(spec['threads']//32)*2*16*spec['k_staging']*2,
                output_bytes=spec['tiles']*256*4,virtual_groups=groups,
                nonempty_ctas=min(grid,groups),work_hash=work_hash)

def validate_outputs(spec,actual):
    """Actual must be a CPU array transferred after the GPU end event, not a stamp."""
    import numpy as np
    x=np.asarray(actual)
    if not np.all(np.isfinite(x)):raise ValueError('nonfinite probe output')
    if spec['probe']=='fp32_fma':
        x=x.reshape(spec['items'],16)
        ref=np.asarray(fma_oracle(spec['rounds']),dtype=np.float32)
        indices=np.arange(x.size,dtype=np.int64)%64
        ok=np.array_equal(x.reshape(-1),ref[indices])
    else:
        x=x.reshape(spec['tiles'],16,16)
        ref=np.asarray(tensor_oracle(spec['mma_rounds']),dtype=np.float32)
        ok=np.allclose(x,ref[None,:,:],rtol=0,atol=1e-4)
    if not ok:raise ValueError('probe result differs from CPU oracle')
    return dict(correctness_passed=True,elements_checked=int(x.size))
