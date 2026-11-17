"""Measured-data figures only. No CPU fixtures or predicted GPU measurements."""
from pathlib import Path
import os,csv,json,gzip,sys
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'local_cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0,str(ROOT/'src'))
from common import union


def activities(stem):
    with gzip.open(ROOT/'analysis'/(stem+'_activities.json.gz'),'rt',encoding='utf-8') as f:return json.load(f)


def main():
    data=json.loads((ROOT/'analysis/gate2_report_data.json').read_text())
    assert len(data['p0'])==4
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':170})
    colors={'bf16':'#3568a0','nvfp4':'#e88b28'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,exp in zip(axes,['E1','E2']):
        for x,prec in enumerate(['bf16','nvfp4']):
            p=next(r for r in data['p0'] if r['experiment']==exp and r['precision']==prec)
            ax.bar(x,p['full_ms']/1000,color=colors[prec],width=.6)
            ax.errorbar(x,p['full_ms']/1000,yerr=np.array([[p['full_ms']-p['process_min_mean_ms']],[p['process_max_mean_ms']-p['full_ms']]])/1000,fmt='none',ecolor='black',capsize=5)
            ax.text(x,p['full_ms']/1000+.10,f"{p['full_ms']/1000:.3f} s",ha='center')
        ax.set(xticks=[0,1],xticklabels=['BF16','NVFP4'],title=f'{exp}: batch '+('1' if exp=='E1' else '8'),ylabel='Full offline inference (s)',ylim=(0,4.6))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('2048-token distinct prompts + 257 outputs; CUDA Graph; prefix cache off')
    fig.text(.5,.02,'P0, no profiler. Error bars: range of independent-process means, not a population confidence interval.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'gate2_p0_full_latency.png');plt.close(fig)

    fig,axes=plt.subplots(4,1,figsize=(12,8.5))
    phase_colors={'PREFILL':'#47976f','DECODE':'#3d6dab','UNKNOWN':'#aa5c7e','MIXED':'#e5a43e'}
    for ax,(exp,prec) in zip(axes,[('E1','bf16'),('E1','nvfp4'),('E2','bf16'),('E2','nvfp4')]):
        path=next(p for p in (ROOT/'analysis').glob(f'gate2_{exp.lower()}_{prec}_p2_*_timeline.json'))
        stem=path.name.removesuffix('_timeline.json');meta=json.loads(path.read_text());rows=activities(stem);base=meta['cpu_full_range']['start']
        for y,phase in enumerate(['PREFILL','DECODE','UNKNOWN']):
            spans=union([(r['start'],r['end']) for r in rows if r['stage']==phase])
            segments=[[(max(0,s-base)/1e9,y),((e-base)/1e9,y)] for s,e in spans]
            if segments:ax.add_collection(LineCollection(segments,colors=phase_colors[phase],linewidths=5))
        ax.set(xlim=(0,(meta['cpu_full_range']['end']-base)/1e9),ylim=(-.5,2.5),yticks=[0,1,2],yticklabels=['Prefill','Decode','Unmapped'],title=f'{exp} {prec.upper()} — independent node capture',xlabel='Trace time from FULL_INFERENCE start (s)')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Actual kernel interval unions by observed stage; no graph envelopes used as busy time')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'gate2_gpu_timeline.png');plt.close(fig)

    metrics=['SMs Active [Throughput %]','Tensor Active [Throughput %]','DRAM Read Bandwidth [Throughput %]']
    fig,axes=plt.subplots(4,3,figsize=(15,10),sharey=True)
    for row,(exp,prec) in enumerate([('E1','bf16'),('E1','nvfp4'),('E2','bf16'),('E2','nvfp4')]):
        path=next(p for p in (ROOT/'analysis').glob(f'gate2_{exp.lower()}_{prec}_p1_*_timeline.json'))
        stem=path.name.removesuffix('_timeline.json');meta=json.loads(path.read_text());base=meta['cpu_full_range']['start'];end=meta['cpu_full_range']['end']
        binned={m:defaultdict(list) for m in metrics}
        with gzip.open(ROOT/'analysis'/(stem+'_metric_samples.csv.gz'),'rt',encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                if r['metric'] not in binned:continue
                t=int(r['timestamp'])
                if base<=t<=end:binned[r['metric']][(t-base)//1000000].append(float(r['value']))
        for col,m in enumerate(metrics):
            ax=axes[row,col];keys=sorted(binned[m]);ax.plot(np.array(keys)/1000,[np.mean(binned[m][k]) for k in keys],lw=.65,color=colors[prec])
            ax.set(title=f'{exp} {prec.upper()} | '+m.split(' [')[0],ylim=(0,102),xlabel='Own P1 trace time (s)',ylabel='Exported % of peak');ax.grid(alpha=.2)
    fig.suptitle('Device-level GPU Metrics: raw 10 kHz; displayed as 1 ms bin means; separate P1 captures')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'gate2_resource_timeseries.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,exp in zip(axes,['E1','E2']):
        for x,prec in enumerate(['bf16','nvfp4']):
            p=next(r for r in data['p0'] if r['experiment']==exp and r['precision']==prec)
            ax.bar(x,p['ttft_ms'],color='#69aa8a',label='TTFT' if x==0 else None)
            ax.bar(x,p['tpot_ms']*256,bottom=p['ttft_ms'],color='#3568a0',label='Output 1 → 257 span' if x==0 else None)
        ax.set(xticks=[0,1],xticklabels=['BF16','NVFP4'],title=exp,ylabel='Mean per-request engine intervals (ms)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('CPU-observed request components; TTFT includes scheduling, prefill and first-output handling')
    fig.text(.5,.015,'These are request intervals, not GPU operator times. Client network and tokenization are excluded.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'gate2_request_components.png');plt.close(fig)
    print('Four measured-data figures saved')


if __name__=='__main__':main()
