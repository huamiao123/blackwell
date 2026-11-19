"""Plot measured E2 P1 device windows and P0 matched completion ratios."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import ROOT

def main():
    folder=ROOT/'analysis/e2_dram'
    def read(n):
        with (folder/n).open(newline='') as f:return list(csv.DictReader(f))
    windows=read('device_windows.csv')
    bounds=read('statistics.csv')
    retention=read('probe_competition.csv')
    models=['bf16','nvfp4']
    labels=['Graph alone\nfirst 500 µs','Graph + read\nfirst 500 µs','Read only\nprobe window','Read + graph\nprobe window']
    keys=[('serial','graph_first_500us'),('parallel','graph_first_500us'),
          ('serial','probe_before_graph'),('parallel','probe_inside_graph')]
    figure,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained',gridspec_kw={'width_ratios':[1.7,1,1]})
    for i,p in enumerate(models):
        color=('#4778a4','#bd6945')[i]
        pct=[]
        for condition,window in keys:
            value,=[float(r['mean_pct']) for r in windows if (r['precision'],r['condition'],r['window'],r['metric'])
                    ==(p,condition,window,'DRAM Read Bandwidth')]
            pct.append(value)
        axes[0].bar([x+(i-.5)*.36 for x in range(4)],pct,width=.35,color=color,label=p.upper())
        for j,profile in enumerate(['P2','P1']):
            v,=[float(r['conditional_retention'])*100 for r in retention if r['precision']==p and r['profile']==profile]
            axes[1].scatter(j+(i-.5)*.10,v,color=color,s=45)
        match,=[r for r in bounds if r['precision']==p and r['metric']=='serial_parallel_joint_ratio']
        v,lo,hi=[float(match[k]) for k in ['mean','ci95_low','ci95_high']]
        axes[2].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4)
    axes[0].set(xticks=range(4),xticklabels=labels,ylim=(0,100),ylabel='Device DRAM read activity (%)',title='10 kHz hardware samples')
    axes[0].tick_params(axis='x',labelsize=8)
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=[0,1],xticklabels=['P2','P1'],xlim=(-.5,1.5),ylim=(75,95),
                ylabel='Probe throughput retained (%)',title='Equal 12 GiB read workload')
    axes[2].axvline(1,color='grey',lw=1)
    axes[2].set(yticks=[0,1],yticklabels=['BF16','NVFP4'],ylim=(-.6,1.6),xlabel='Measured serial / parallel',title='Full-request P0 joint time')
    axes[2].set_xticks([1.000,1.007,1.014],['1.000','1.007','1.014'])
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    figure.savefig(ROOT/'figures/full_decode_dram.png',dpi=175)
    plt.close(figure)
if __name__=='__main__':
    main()
