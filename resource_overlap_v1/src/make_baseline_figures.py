"""Publication-style baseline plots; never fabricate missing overlap results."""
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'local_cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'figures';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((root/'analysis/baseline.csv').open()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    colors={'bf16':'#386cb0','nvfp4':'#e68522'}
    for ax,batch in zip(axes,[1,8]):
        for x,prec in enumerate(['bf16','nvfp4']):
            rr=[r for r in rows if int(r['batch'])==batch and r['precision']==prec]
            vals=np.array([float(r['mean_step_ms']) for r in rr])
            ax.bar(x,vals.mean(),width=.56,color=colors[prec],alpha=.78)
            ax.scatter(x+np.linspace(-.12,.12,len(vals)),vals,color='black',s=15,zorder=3)
            ax.text(x,vals.mean()+.28,f'{vals.mean():.3f} ms',ha='center')
        ax.set(title=f'Batch {batch}',xticks=[0,1],xticklabels=['BF16','NVFP4'],ylim=(0,13))
        ax.set_ylabel('Decode time per step (ms)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fair CUDA Graph Decode: 2048-token distinct prompts, prefix caching off')
    fig.text(.5,.015,'5 repeated generations per configuration; 256 Decode steps; no profiler. Dots are trial means.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    for ext in ['png','svg']:fig.savefig(out/f'baseline_latency.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True)
    for ax,(batch,prec) in zip(axes.flat,[(1,'bf16'),(1,'nvfp4'),(8,'bf16'),(8,'nvfp4')]):
        file=next((root/'raw').glob(f'gate1_{prec}_b{batch}_s*.json'))
        d=json.loads(file.read_text())
        for r in d['runs']:
            ax.plot([x['step'] for x in r['steps']],[x['latency_ms'] for x in r['steps']],lw=.7,alpha=.5)
        ax.set_title(f'Batch {batch}, {prec.upper()}');ax.set_ylabel('Output-completion interval (ms)')
        ax.grid(alpha=.2)
    for ax in axes[-1,:]:ax.set_xlabel('Decode step')
    fig.suptitle('CPU-visible per-step completion intervals (five repeated generations)')
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','svg']:fig.savefig(out/f'baseline_step_series.{ext}')
    plt.close(fig)
    print('Saved baseline figures; overlap figures pending GPU experiments.')


if __name__=='__main__':main()
