"""Finite probes, independent capture, output checks, compiler evidence."""
from pathlib import Path
import ctypes
import hashlib
import json
import subprocess
import struct
import torch
import triton
import triton.language as tl


@triton.jit
def _aux_empty(Stamp, CHUNK):
    tl.store(Stamp + CHUNK, CHUNK + 1)


class Probe:
    def __init__(self, kind, chunks=128, reps=800000, inner=512, ctas=8):
        self.kind, self.chunks, self.reps, self.inner, self.ctas = kind, chunks, reps, inner, ctas
        self.stamp = torch.zeros(chunks, device='cuda', dtype=torch.int32)
        self.config = dict(kind=kind, chunks=chunks, reps=reps, inner=inner, ctas=ctas)
        self.kernel = None
        if kind == 'fp32':
            self.x = torch.arange(ctas * 128, device='cuda', dtype=torch.float32) / 8192 + .125
            self.y = torch.empty((chunks, ctas * 128, 4), device='cuda', dtype=torch.float32)
            self.work = dict(flops=chunks * ctas * 128 * 4 * reps * 2,
                             logical_bytes=chunks * ctas * 128 * 5 * 4)
        elif kind == 'dram':
            self.x = torch.arange(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.int32)
            self.y = torch.empty_like(self.x)
            self.work = dict(flops=0, logical_bytes=self.x.numel() * 4 * 2 * chunks)
        elif kind == 'tensor':
            # Deterministic integer-valued BF16 inputs; exact CPU reference.
            self.a = (((torch.arange(64 * 1024, device='cuda') * 17 + 3) % 7) - 3).reshape(64, 1024).to(torch.bfloat16)
            self.b = (((torch.arange(1024 * 128, device='cuda') * 13 + 5) % 11) - 5).reshape(1024, 128).to(torch.bfloat16)
            self.y = torch.empty((chunks, 64, 128), device='cuda', dtype=torch.float32)
            self.work = dict(flops=2 * 64 * 128 * 1024 * inner * chunks,
                             logical_bytes=None)
        else:
            assert kind == 'empty'
            self.work = dict(flops=0, logical_bytes=chunks * 4)
        self.stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(self.stream):
            self.launch(0)
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            for c in range(chunks):
                self.launch(c)
        self.stream.synchronize()

    def launch(self, chunk):
        if self.kind == 'fp32':
            from auxiliary_fp32 import launch
            k = launch(self.x, self.y, self.stamp, self.reps, chunk, self.ctas)
        elif self.kind == 'dram':
            from auxiliary_dram import launch
            k = launch(self.x, self.y, self.stamp, chunk)
        elif self.kind == 'tensor':
            from auxiliary_tensor import launch
            k = launch(self.a, self.b, self.y, self.stamp, self.inner, chunk)
        else:
            k = _aux_empty[(1,)](self.stamp, chunk, num_warps=1)
        self.kernel = k

    def reset(self):
        self.stamp.zero_()
        if hasattr(self, 'y'):
            self.y.fill_(-777)
        torch.cuda.synchronize()

    def evidence(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        k = self.kernel
        meta = {key: getattr(k.metadata, key, None) for key in
                ['name', 'num_warps', 'shared', 'num_ctas', 'maxnreg']}
        meta.update(n_regs=getattr(k, 'n_regs', None), n_spills=getattr(k, 'n_spills', None))
        ptx = k.asm['ptx']
        (directory / (self.kind + '.ptx')).write_text(ptx)
        cubin = directory / (self.kind + '.cubin')
        cubin.write_bytes(k.asm['cubin'])
        r = subprocess.run(['/usr/local/cuda-13.3/bin/cuobjdump', '--dump-sass', str(cubin)],
                           capture_output=True, text=True, timeout=30)
        (directory / (self.kind + '.sass')).write_text(r.stdout + r.stderr)
        meta.update(ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(), sass_returncode=r.returncode,
                    has_fp32_fma='fma.rn.f32' in ptx,
                    has_bf16_mma='mma.sync' in ptx and 'bf16' in ptx)
        if self.kind == 'fp32':
            assert meta['has_fp32_fma'], 'FP32 instruction check failed'
        if self.kind == 'tensor':
            assert meta['has_bf16_mma'], 'Not a verified BF16 Tensor Core probe'
        return meta

    def validate(self, cache):
        stamps = self.stamp.cpu().tolist()
        assert stamps == list(range(1, self.chunks + 1)), ('missing work', stamps)
        info = dict(stamps_checked=len(stamps))
        if self.kind == 'dram':
            assert torch.equal(self.x, self.y), 'DRAM copy mismatch'
            info.update(elements_checked=self.x.numel(), full_output_check=True)
        elif self.kind == 'tensor':
            ref = (self.a.float().cpu().to(torch.int64) @ self.b.float().cpu().to(torch.int64)) * self.inner
            # Bound ensures integer accumulation remains exactly representable.
            assert 1024 * 15 * self.inner < 2**24
            yy = self.y.cpu()
            assert torch.equal(yy, ref.float().expand_as(yy)), 'BF16 MMA output mismatch'
            info.update(elements_checked=yy.numel(), max_error=0, full_output_check=True)
        elif self.kind == 'fp32':
            cache = Path(cache)
            cache.mkdir(parents=True, exist_ok=True)
            src = cache / 'reference_fma.c'
            so = cache / 'reference_fma.so'
            code = '''#include <math.h>\n#include <stdint.h>\n#include <string.h>\nvoid ref(float x,int n,float *o){uint32_t u[4]={0x3f800001,0x3f800002,0x3f7ffffd,0x3f7ffff9};float m[4],v[4]={x,x+.125f,x+.25f,x+.375f};memcpy(m,u,16);uint32_t q=0x33d6bf95;float z;memcpy(&z,&q,4);for(int i=0;i<n;i++)for(int j=0;j<4;j++)v[j]=fmaf(v[j],m[j],z);memcpy(o,v,16);}\n'''
            if not src.exists() or src.read_text() != code or not so.exists():
                src.write_text(code)
                subprocess.run(['gcc', '-O2', '-fPIC', '-shared', str(src), '-o', str(so), '-lm'], check=True, timeout=30)
            lib = ctypes.CDLL(str(so))
            lib.ref.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            ix = [0, 1, self.ctas * 128 // 2, self.ctas * 128 - 1]
            refs = []
            for i in ix:
                buf = (ctypes.c_float * 4)()
                lib.ref(float(self.x[i].item()), self.reps, buf)
                refs.append(list(buf))
            yy = self.y[:, ix, :].cpu()
            ref = torch.tensor(refs).expand_as(yy)
            assert torch.equal(yy, ref), ('FP32 recurrence mismatch', (yy - ref).abs().max().item())
            assert bool(torch.isfinite(self.y).all()), 'nonfinite FMA output'
            info.update(sampled_lanes=ix, chunks_checked=self.chunks, max_error=0,
                        full_output_check=False, reference='CPU C fmaf, exact sampled recurrence')
        info['passed'] = True
        return info

    def describe(self):
        return dict(config=self.config, work=self.work, stream_handle=int(self.stream.cuda_stream))
