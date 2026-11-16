from pathlib import Path
import json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent/'delivery';F=ROOT/'figures';F.mkdir(parents=True,exist_ok=True)
bundle=json.loads((ROOT/'report_data.json').read_text())
S=[r for r in bundle['summaries'] if r['protocol']=='default_scheduler_v2']
def pick(batch,precision,mode,kind='metrics',rep=2):
    match=[s for s in S if s['args']['batch']==batch and s['args']['precision']==precision and s['args']['mode']==mode and f'_{kind}_' in s['trace'] and s['trace'].endswith(f'r{rep:02d}')]
    return match[0] if match else None
colors={'bf16':'#2563a6','nvfp4':'#ca572b'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
batches=sorted({r['args']['batch'] for r in S if '_metrics_' in r['trace']},reverse=True)
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)
# 1. Graph envelopes and observed regular kernels are visibly distinct.
fig,axes=plt.subplots(4,len(batches),figsize=(8*len(batches),9),squeeze=False)
for col,b in enumerate(batches):
    for row,(prec,mode) in enumerate([('bf16','eager'),('nvfp4','eager'),('bf16','graph'),('nvfp4','graph')]):
        ax=axes[row,col];s=pick(b,prec,mode)
        if not s:ax.text(.5,.5,'Not collected',transform=ax.transAxes,ha='center');continue
        for a,z in s['plot']['short_graphs']:ax.broken_barh([(a/1e6,(z-a)/1e6)],(.15,.65),facecolors='#bcd8ae',edgecolors='#5b8646',alpha=.8)
        for a,z in s['plot']['short_kernels']:ax.broken_barh([(a/1e6,(z-a)/1e6)],(.25,.45),facecolors=colors[prec],linewidth=0)
        ax.set(xlim=(0,25),ylim=(0,1),yticks=[],title=f'B{b} {prec.upper()} {mode} (10 kHz metrics)')
        ax.set_xlabel('ms from steady-window start')
        if mode=='graph':ax.text(.02,.92,'Green = graph envelope; kernel coverage unavailable here',transform=ax.transAxes,fontsize=8,va='top')
fig.suptitle('Observed GPU timeline: graph envelopes are not kernel busy time',fontsize=14)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'01_gpu_timeline')
# 2. Time series use actual elapsed time; display is 1-ms bin means of 100-us samples.
fig,axes=plt.subplots(len(batches),2,figsize=(14,3.5*len(batches)),squeeze=False)
for row,b in enumerate(batches):
    for col,mode in enumerate(['eager','graph']):
        ax=axes[row,col]
        for prec in ['bf16','nvfp4']:
            s=pick(b,prec,mode)
            if not s:continue
            series=s['plot']['series'];ax.plot([r['time_ms'] for r in series],[r.get('DRAM Read Bandwidth [Throughput %]',np.nan) for r in series],color=colors[prec],label=prec.upper(),lw=.8)
        ax.set(title=f'B{b}, {mode}; measured under NSYS',xlabel='ms in steady Decode window',ylabel='DRAM read (% of peak)',ylim=(0,100));ax.legend()
fig.suptitle('DRAM read: 1-ms display bins; raw 10-kHz samples preserved',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'02_dram_read_timeseries')
# 3. Distinct panels for SM activity and Tensor activity, without treating them as FLOPs.
fig,axes=plt.subplots(2*len(batches),2,figsize=(14,3*2*len(batches)),squeeze=False)
for bi,b in enumerate(batches):
    for mi,mode in enumerate(['eager','graph']):
        for col,(metric,title) in enumerate([('SMs Active [Throughput %]','SMs Active'),('Tensor Active [Throughput %]','Tensor Active')]):
            ax=axes[2*bi+mi,col]
            for prec in ['bf16','nvfp4']:
                s=pick(b,prec,mode)
                if s:
                    series=s['plot']['series'];ax.plot([r['time_ms'] for r in series],[r.get(metric,np.nan) for r in series],color=colors[prec],label=prec.upper(),lw=.8)
            ax.set(title=f'B{b} {mode}: {title}',xlabel='ms in steady window',ylabel='% of peak',ylim=(0,100));ax.legend()
fig.suptitle('Device-level sampled activity under NSYS (not usable-resource fractions)',fontsize=14)
fig.tight_layout(rect=(0,0,1,.97));save(fig,'03_sm_tensor_timeseries')
# 4. Decomposition stays inside each profiled run. Normal baseline is a separate panel.
fig,axes=plt.subplots(len(batches),2,figsize=(14,4*len(batches)),squeeze=False)
for row,b in enumerate(batches):
    labels=[];active=[];gaps=[]
    for prec,mode,kind in [('bf16','eager','metrics'),('nvfp4','eager','metrics'),('bf16','graph','node'),('nvfp4','graph','node')]:
        s=pick(b,prec,mode,kind)
        if s and s['kernel_coverage_reliable']:
            n=s['analyzed_steps'];labels.append(prec.upper()+'\n'+mode+('\nnode/32' if kind=='node' else '\nmetrics/224'))
            active.append(s['kernel_union_ms']/n);gaps.append(s['gap_ms']/n)
    ax=axes[row,0];x=np.arange(len(labels));ax.bar(x,active,label='Observed kernel union',color='#3a7d9c');ax.bar(x,gaps,bottom=active,label='No observed kernel',color='#d5ad68')
    ax.set(xticks=x,xticklabels=labels,ylabel='ms / analyzed Decode step',title=f'B{b}: within-profile time decomposition');ax.legend(fontsize=8)
    ax=axes[row,1]
    x=np.arange(2);width=.32
    for j,mode in enumerate(['eager','graph']):
        vals=[]
        for prec in ['bf16','nvfp4']:
            vals.append(statistics.median([r['mean_step_ms'] for d in bundle['raw_briefs'] if d.get('protocol')=='default_scheduler_v2' and '_baseline_' in d['file'] and d['args']['batch']==b and d['args']['precision']==prec and d['args']['mode']==mode for r in d['runs']]))
        ax.bar(x+(j-.5)*width,vals,width,label=mode,color=['#6d8eac','#76a67d'][j])
    ax.set(xticks=x,xticklabels=['BF16','NVFP4'],ylabel='ms / Decode step',title=f'B{b}: normal baseline (5 requests/config)');ax.legend()
fig.suptitle('Profiled decompositions and unprofiled performance use separate panels',fontsize=14)
fig.tight_layout(rect=(0,0,1,.95));save(fig,'04_decode_time_breakdown')
print('FIGURES',*[str(x) for x in sorted(F.glob('*.png'))],sep='\n')
