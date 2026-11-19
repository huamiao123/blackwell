"""Compile and inspect the DRAM probe CUBIN without loading CUDA."""

import argparse
from pathlib import Path
import re
import subprocess

from compile_probes import sections
from g4_common import save, sha, write_csv


def read_instructions(sass):
    return {
        name: {opcode: len(re.findall(rf'\b{opcode}(?:\.|\s)', body))
               for opcode in ('LDG', 'STG', 'LDS', 'STS', 'LDL', 'STL')}
        for name, body in sections(sass)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--cuda-bin', type=Path, default=Path('/usr/local/cuda-13.3/bin'))
    args = parser.parse_args()
    build = args.build.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('dram_streaming.cu')
    binary = build / 'dram_streaming_sm120.cubin'
    nvcc, cuobjdump = (str(args.cuda_bin / name) for name in ('nvcc', 'cuobjdump'))
    command = [nvcc, '-arch=sm_120', '-std=c++17', '-O3', '-lineinfo', '--cubin', str(source), '-o', str(binary)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (build / 'nvcc.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    process.check_returncode()
    resource_text = subprocess.check_output([cuobjdump, '--dump-resource-usage', str(binary)], text=True, timeout=30)
    sass_text = subprocess.check_output([cuobjdump, '--dump-sass', str(binary)], text=True, timeout=30)
    sass_path = build / 'sass.txt'
    sass_path.write_text(sass_text, encoding='utf-8', newline='\n')
    (build / 'resources.txt').write_text(resource_text, encoding='utf-8', newline='\n')
    resources = dict(sections(resource_text))
    instructions = read_instructions(sass_text)
    rows = []
    for operation in ('read', 'write', 'copy'):
        for variant, threads, dynamic in [('light', 32, 0), ('heavy', 128, 32768)]:
            symbol = f'dram_{operation}_{variant}'
            stats = instructions[symbol]
            if (not stats['STG'] or bool(stats['LDG']) != (operation != 'write') or stats['LDL'] or stats['STL']
                    or (variant == 'heavy' and (not stats['LDS'] or not stats['STS']))):
                raise ValueError(f'unexpected instructions for {symbol}: {stats}')
            body = resources[symbol]
            rows.append(dict(symbol=symbol, operation=operation, variant=variant, threads_per_cta=threads,
                             registers_per_thread=int(re.search(r'REG:(\d+)', body)[1]),
                             static_shared_bytes=int(re.search(r'SHARED:(\d+)', body)[1]),
                             planned_dynamic_shared_bytes=dynamic, **stats, source_sha256=sha(source),
                             cubin_sha256=sha(binary), sass_sha256=sha(sass_path),
                             gpu_correctness='UNVERIFIED', physical_dram_bytes='UNAVAILABLE',
                             status='OFFLINE_COMPILED_NOT_GPU_EXECUTED'))
    write_csv(build / 'dram_probe_validation.csv', rows)
    save(build / 'compile_manifest.json', dict(command=command, nvcc_version=subprocess.check_output(
        [nvcc, '--version'], text=True, timeout=15), cpu_only=True, gpu_module_loaded=False,
        source_sha256=sha(source), cubin_sha256=sha(binary)))
    print(f'Inspected {len(rows)} DRAM probe symbols, no CUDA runtime used')


if __name__ == '__main__':
    main()
