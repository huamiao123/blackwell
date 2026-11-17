"""Offline nvcc/cuobjdump only. No CUDA initialization, module load or launch."""
from pathlib import Path
import argparse,re,subprocess
from g4_common import *

SYMBOLS=['fp32_light','fp32_heavy','tensor_light','tensor_heavy']

def sections(text):
    matches=list(re.finditer(r'^[ \t]*Function[ \t]*:?[ \t]*(\w+)[ \t]*:?[ \t]*$',text,re.M))
    return [(m[1],text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]) for i,m in enumerate(matches)]

def parse_resources(text):
    result={}
    for name,body in sections(text):
        m=re.search(r'REG:(\d+)',body)
        if name in SYMBOLS and m:result[name]=dict(registers_per_thread=int(m[1]),
            static_shared_bytes=int(re.search(r'SHARED:(\d+)',body)[1]) if re.search(r'SHARED:(\d+)',body) else 0)
    return result

def parse_sass(text):
    result={}
    for name,body in sections(text):
        if name not in SYMBOLS:continue
        result[name]=dict(ffma_static_instructions=len(re.findall(r'\bFFMA\b',body)),
                         bf16_mma_static_instructions=len(re.findall(r'\b(?:HMMA|MMA)[^;\n]*BF16',body)),
                         tensor_mnemonics=sorted(set(re.findall(r'\b(?:HMMA|MMA)[A-Z0-9_.]*',body))))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--nvcc',default='/usr/local/cuda-13.3/bin/nvcc')
    p.add_argument('--cuobjdump',default='/usr/local/cuda-13.3/bin/cuobjdump');p.add_argument('--build',required=True)
    a=p.parse_args();build=Path(a.build).resolve();build.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('probes.cu');cubin=build/'probes_sm120.cubin'
    command=[a.nvcc,'-arch=sm_120','-std=c++17','-O3','-lineinfo','--cubin',str(source),'-o',str(cubin)]
    r=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (build/'nvcc.log').write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError('offline compilation failed; see nvcc.log')
    res=subprocess.run([a.cuobjdump,'--dump-resource-usage',str(cubin)],capture_output=True,text=True,timeout=40)
    sass=subprocess.run([a.cuobjdump,'--dump-sass',str(cubin)],capture_output=True,text=True,timeout=40)
    if res.returncode or sass.returncode:raise RuntimeError('offline disassembly failed')
    (build/'resources.txt').write_text(res.stdout,encoding='utf-8');(build/'sass.txt').write_text(sass.stdout,encoding='utf-8')
    resources=parse_resources(res.stdout);instructions=parse_sass(sass.stdout)
    rows=[]
    for symbol in SYMBOLS:
        if symbol not in resources or symbol not in instructions:raise ValueError('missing compiled function '+symbol)
        ins=instructions[symbol]
        # cuobjdump may spell BF16 type qualifiers differently. Missing explicit
        # BF16 confirmation is UNVERIFIED, never inferred from tensor activity.
        status='OFFLINE_COMPILED_NOT_GPU_EXECUTED'
        rows.append(dict(symbol=symbol,**resources[symbol],**ins,compile_status=status,
                         gpu_correctness='UNVERIFIED',standalone_latency='UNAVAILABLE',
                         sass_sha256=sha(build/'sass.txt'),cubin_sha256=sha(cubin),source_sha256=sha(source)))
    write_csv(build/'probe_validation.csv',rows)
    save(build/'compile_manifest.json',dict(cpu_only=True,gpu_module_loaded=False,command=command,source_sha256=sha(source),
         cubin_sha256=sha(cubin),nvcc_version=subprocess.check_output([a.nvcc,'--version'],text=True),
         note='SASS counts are static assembly evidence, not dynamic instruction counts or GPU correctness'))
    print('Offline compiled/disassembled four probes. No CUDA workload launched.')

if __name__=='__main__':main()
