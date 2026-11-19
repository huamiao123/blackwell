"""Plot per-process probe retention and device activity on matched E2 stages."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def plot(root, figure):
    comparisons=read(root/'phase_comparison.csv')
    metrics=read(root/'device_metrics.csv')
    fig, axes=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
    colors={'nvfp4':'#00736c','bf16':'#77539b'}
    for precision,offset in [('nvfp4',-.11),('bf16',.11)]:
        for cohort,marker in [('original','o'),('independent_repeat','s')]:
            rows={r['stage']:float(r['conditional_throughput_ratio'])*100 for r in comparisons
                  if r['precision']==precision and r['cohort']==cohort}
            axes[0].scatter(np.arange(2)+offset,[rows[s] for s in ('PREFILL','DECODE')], marker=marker,
                            facecolor=colors[precision] if cohort=='independent_repeat' else 'white',
                            edgecolor=colors[precision],s=65,
                            label=f'{precision.upper()} / '+('first' if cohort=='original' else 'repeat'))
    axes[0].set(xticks=[0,1],xticklabels=['Prefill','Decode'],xlim=(-.4,1.4),ylabel='Probe throughput retained (%)',
                title='Tensor probe vs. stage serial')
    axes[0].legend(fontsize=8,loc='center left')
    for ax,stage,title in zip(axes[1:],('prefill_gpu_span','decode_33_64_graphs'),('Prefill GPU span','Decode replays 33–64')):
        for precision,offset in [('bf16',-.18),('nvfp4',.18)]:
            sel={r['metric'].split(' [')[0]:float(r['mean_pct']) for r in metrics if r['condition']=='a_alone'
                 and r['precision']==precision and r['stage']==stage}
            ax.bar(np.arange(3)+offset,[sel[m] for m in ('SMs Active','Tensor Active','DRAM Read Bandwidth')],
                   width=.34,color=colors[precision],label=precision.upper())
        ax.set(xticks=range(3),xticklabels=['SMs active','Tensor active','DRAM read'],title=title,ylabel='Device metric (% of peak)')
        ax.legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(0,105)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(figure,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    plot(Path(sys.argv[1]),Path(sys.argv[2]))
