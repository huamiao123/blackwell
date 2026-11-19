"""Render the observed target windows and independent P0 block estimates."""
import csv
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
def rows(name):
    with (ROOT/'analysis'/name).open() as f:return list(csv.DictReader(f))

def main():
    OUT.mkdir(exist_ok=True)
    nodes=rows('p2_nodes.csv')
    forecast=rows('forecast_check.csv')
    cases=[('nvfp4','o'),('nvfp4','down'),('bf16','o'),('bf16','down')]
    labels=['NVFP4 O','NVFP4 Down','BF16 O','BF16 Down']
    fig,(timeline,points)=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for i,(prec,op) in enumerate(cases):
        actual=[r for r in nodes if r['precision']==prec and r['mode']==op+'_parallel']
        center=median(float(r['finish_margin_ns']) for r in actual)
        r=min(actual,key=lambda v:(abs(float(v['finish_margin_ns'])-center),v['run_id'],v['step']))
        width=float(r[op+'_ns'])/1000
        left=float(r['start_delta_ns'])/1000
        timeline.barh(i-.13,width,height=.22,color='#3d6687',label='Model projection' if i==0 else None)
        timeline.barh(i+.13,float(r['b_ns'])/1000,left=left,height=.22,color='#cf8449',label='Tensor probe' if i==0 else None)
        rr=[r for r in forecast if (r['precision'],r['operator'])==(prec,op)]
        points.scatter(float(rr[0]['forecast_margin_us']),i+.16,marker='D',facecolor='none',edgecolor='#253b49',s=55,label='Earlier duration estimate' if i==0 else None)
        for j,v in enumerate(rr):
            points.scatter(float(v['observed_median_margin_us']),i-.08-.13*j,marker='o',color='#bd6b3d',s=30,
                           label='New process median' if i==0 and j==0 else None)
    for ax in (timeline,points):
        ax.set(yticks=range(4),yticklabels=labels,ylim=(4.25,-.55))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.legend(loc='lower right',fontsize=8)
    timeline.set(xlabel='GPU time relative to target start (µs)',title='Same workload inside the real Decode graph')
    points.axvline(0,color='#52626b',lw=1)
    points.set(xlabel='Target end minus probe end (µs)',title='Does the probe finish inside the target?')
    fig.savefig(OUT/'target_windows.png',dpi=175);plt.close(fig)
    summary=ROOT/'analysis/p0_summary.csv'
    if not summary.exists():return
    ss=rows('p0_summary.csv')
    f,(ax,bx)=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,(prec,op) in enumerate(cases):
        for j,rep in enumerate(['r1','r2']):
            for k,metric in enumerate(['serial_ab_over_parallel','serial_ba_over_parallel']):
                group=[r for r in ss if (r['precision'],r['op'],r['rep'],r['metric'])==(prec,op,rep,metric)]
                if not group:continue
                x=group[0];v,lo,hi=[100*(float(x[key])-1) for key in ('mean','ci95_low','ci95_high')]
                ax.errorbar(v,i+(2*j+k-1.5)*.135,xerr=[[v-lo],[hi-v]],fmt='o' if j==0 else 's',
                            color=('#3d6687','#bd6b3d')[k],ms=4,capsize=2,
                            label=(rep.upper()+' '+('AB','BA')[k]) if i==0 else None)
    for i,prec in enumerate(['nvfp4','bf16']):
        for j,rep in enumerate(['r1','r2']):
            group=[r for r in ss if (r['precision'],r['rep'],r['metric'])==(prec,rep,'down_minus_o_parallel_joint_ms')]
            if not group:continue
            v,lo,hi=[float(group[0][key]) for key in ('mean','ci95_low','ci95_high')]
            bx.errorbar(v,i+(j-.5)*.15,xerr=[[v-lo],[hi-v]],fmt='o' if j==0 else 's',color='#3d6687',ms=5,capsize=3,
                        label=rep.upper() if i==0 else None)
    ax.set(yticks=range(4),yticklabels=labels,ylim=(4.3,-.55),xlabel='(Serial / parallel − 1) × 100 (%)',title='Whole-request joint completion')
    bx.set(yticks=[0,1],yticklabels=['NVFP4','BF16'],ylim=(1.6,-.55),xlabel='Down placement minus O placement (ms)',title='Paired placement difference')
    for a in (ax,bx):
        a.axvline(0,color='#718087',lw=1);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
        a.legend(loc='lower right',fontsize=8,ncol=2);a.spines[['right','top']].set_visible(False)
    f.savefig(OUT/'request_pairs.png',dpi=175);plt.close(f)
if __name__=='__main__':main()
