"""Read saved supervisor logs; never contact CUDA or nvidia-smi."""
import csv
import json
from pathlib import Path
import re
import sys

def number(pattern,s):
    match=re.search(pattern,s,re.M)
    return float(match[1]) if match else ''

def main():
    root=Path.home()/'fp4-llm/results/full_inference_v4/logs'
    fields=['run_id','sample_in_run','compute_processes','clock_sm_mhz','clock_mem_mhz',
            'power_instant_w','gpu_temp_c','monotonic_ns']
    w=csv.writer(sys.stdout,lineterminator='\n')
    w.writerow(fields)
    for rid in sys.argv[1:]:
        if not re.fullmatch(r'micro_[a-z0-9_]+',rid):
            raise ValueError('invalid run ID')
        item=json.loads((root/(rid+'_supervision.json')).read_bytes())
        for i,entry in enumerate(item['telemetry']):
            s=entry['gpu']
            w.writerow([rid,i,len(entry['compute_pids']),
                        number(r'^    Clocks\s*\n(?:        [^\n]+\n){0,2}?        SM\s*:\s*([\d.]+) MHz',s),
                        number(r'^    Clocks\s*\n(?:        [^\n]+\n){0,3}?        Memory\s*:\s*([\d.]+) MHz',s),
                        number(r'^        Instantaneous Power Draw\s*:\s*([\d.]+) W',s),
                        number(r'^        GPU Current Temp\s*:\s*([\d.]+) C',s),entry['monotonic_ns']])

if __name__=='__main__':
    main()
