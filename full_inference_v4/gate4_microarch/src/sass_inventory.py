"""CPU-only import of existing NCU reports on the server."""
import csv
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter

def main():
    folder=Path.home()/'fp4-llm/results/full_inference_v4/gate4_microarch/reports'
    out=csv.writer(sys.stdout,lineterminator='\n')
    out.writerow(['run_id','sass_sha256','opcode','static_instruction_sites'])
    for rid in sys.argv[1:]:
        if not re.fullmatch(r'micro_[a-z0-9_]+',rid):raise ValueError('invalid run ID')
        report=folder/(rid+'.ncu-rep')
        data=subprocess.check_output(['/opt/nvidia/nsight-compute/2026.2.1/ncu','--import',str(report),
                                      '--page','source','--print-source','sass'],timeout=60)
        counts=Counter()
        for ln in data.decode().splitlines():
            m=re.match(r'^0x[0-9a-f]+\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]+)(?=\s|$)',ln)
            if m:counts[m[1]]+=1
        if not counts:raise ValueError('no instruction records')
        digest=hashlib.sha256(data).hexdigest()
        for op,n in sorted(counts.items()):out.writerow([rid,digest,op,n])

if __name__=='__main__':
    main()
