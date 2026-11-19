"""Plot real Q/K completion and whole-request interference from separate measurements."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
def table(name):
    with (ROOT/'analysis'/name).open() as f:return list(csv.DictReader(f))

def main():
    timings=table('p2_summary.csv')
    paired=table('p0_estimates.csv')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    dots={'r1':'o','r2':'s'}
    colors={'sham':'#386184','fork':'#c46f34'}
    for i,precision in enumerate(['nvfp4','bf16']):
        for j,mode in enumerate(['sham','fork']):
            for rep,offset in [('r1',-.045),('r2',.045)]:
                obs=next(r for r in timings if (r['precision'],r['rep'],r['mode'])==(precision,rep,mode))
                ax.scatter(float(obs['pair_us_median']),i+(j-.5)*.34+offset,marker=dots[rep],s=50,color=colors[mode])
        for rep,offset in [('r1',-.085),('r2',.085)]:
            r=next(r for r in paired if (r['precision'],r['rep'],r['control'],r['metric'])==(precision,rep,'sham','model_ms'))
            cent,lo,hi=[float(r[k]) for k in ('pct_diff','pct_diff_ci95_lo','pct_diff_ci95_hi')]
            bx.errorbar(cent,i+offset,xerr=[[cent-lo],[hi-cent]],color='#386184',fmt=dots[rep],capsize=4)
    ax.set(xlabel='Q/K pair span (µs)',title='GPU timeline · median across layers and steps',xlim=(0,None))
    bx.set(xlabel='Fork minus serial clone (%)',title='Unprofiled full-request latency · paired 95% CI')
    bx.axvline(0,color='#697983',lw=1)
    for a in (ax,bx):
        a.set(yticks=(0,1),yticklabels=['NVFP4','BF16'],ylim=(1.65,-.55))
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=c,label=label) for c,label in [(colors['sham'],'Serial clone'),(colors['fork'],'Q/K fork')]],loc='lower right')
    bx.legend(handles=[Line2D([0],[0],marker=m,ls='',color='#386184',label=label) for m,label in [('o','Process 1'),('s','Process 2')]],loc='lower right')
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'qk_completion.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    main()
