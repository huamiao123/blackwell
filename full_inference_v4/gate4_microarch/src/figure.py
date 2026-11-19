"""Resource denominators, real Decode node lengths, and validated PC sampling."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import ROOT

def rows(n):
    with (ROOT/'analysis'/n).open() as f:return list(csv.DictReader(f))

def main():
    cases={(r['precision'],r['operator']):r for r in rows('projection_comparison.csv')}
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    t=axes[0]
    keys=[('bf16','down_proj'),('nvfp4','down_proj'),('bf16','gate_up_proj'),('nvfp4','gate_up_proj')]
    labels=['BF16 Down','NVFP4 Down','BF16 Gate/Up','NVFP4 Gate/Up']
    ix=list(range(len(keys)))
    for field,off,legend,color in [('p3_tensor_elapsed_pct',-.18,'Elapsed cycles','#4778a4'),
                                   ('p3_tensor_active_denominator_pct',.18,'Active cycles','#df9b42')]:
        t.barh([x+off for x in ix],[float(cases[k][field]) for k in keys],height=.34,label=legend,color=color)
    t.set(yticks=ix,yticklabels=labels,xlim=(0,100),xlabel='Tensor pipe activity (%)',title='Same pipe, two denominators')
    t.invert_yaxis();t.legend(loc='lower right',fontsize=8)
    t=axes[1]
    ops=['qkv_proj','o_proj','down_proj','gate_up_proj']
    durations=rows('full_decode_duration_bounds.csv')
    for p,off,c in [('bf16',-.13,'#4778a4'),('nvfp4',.13,'#bc6742')]:
        rr=[next(r for r in durations if r['precision']==p and r['operator']==op and r['layer']=='all') for op in ops]
        q=[float(r['p50_us']) for r in rr]
        t.errorbar(q,[i+off for i in ix],xerr=[[v-float(r['p05_us']) for r,v in zip(rr,q)],
                                                 [float(r['p95_us'])-v for r,v in zip(rr,q)]],
                   fmt='o',capsize=3,label=p.upper(),color=c)
    t.axvspan(min(float(r['isolated_b_alone_us']) for r in durations),max(float(r['isolated_b_alone_us']) for r in durations),
              color='#777777',alpha=.3,label='B alone (isolated)')
    t.set(yticks=ix,yticklabels=['QKV','O','Down','Gate/Up'],xlabel='Full Decode P2 node duration (µs)',
          title='Median and P5–P95, 9,216 nodes')
    t.invert_yaxis();t.legend(fontsize=8)
    t=axes[2];pc=rows('pc_waits.csv')
    for i,k in enumerate(keys):
        matches=[r for r in pc if (r['precision'],r['operator'])==k]
        if not matches:continue
        for j,r in enumerate(matches):
            t.scatter(float(r['sleeping_pct']),i+(j-(len(matches)-1)/2)*.12,
                      color=('#4778a4' if k[0]=='bf16' else '#bc6742'),s=46)
    t.set(yticks=ix,yticklabels=labels,xlim=(-2,100),xlabel='Sleeping samples / all PC samples (%)',
          title='Each point is one P3 process')
    t.invert_yaxis()
    for ax in axes:
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['right','top']].set_visible(False)
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'projection_mechanisms.png',dpi=175)
    plt.close(fig)
if __name__=='__main__':
    main()
