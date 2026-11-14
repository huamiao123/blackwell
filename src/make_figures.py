import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.load(open('raw/interference_full.json'))
def ov(r):
    tA=r['T_A_ms']/r['S_A']; tB=r['T_B_ms']/r['S_B']
    hi,lo=max(tA,tB),min(tA,tB)
    return 0.0 if lo<=0 else max(0.0,min(1.0,1.0-(r['T_joint_ms']-hi)/lo))
for r in rows: r['overlap']=ov(r)
Ms=sorted({r['M'] for r in rows}); probes=['fp32','dram','tensor']
# fig1 heatmap of S_A at k=1
fig,axes=plt.subplots(1,2,figsize=(11,3.6))
for ax,dt in zip(axes,['bf16','fp4']):
    Z=np.zeros((len(probes),len(Ms)))
    for i,p in enumerate(probes):
        for j,m in enumerate(Ms):
            v=[r['S_A'] for r in rows if r['dtype']==dt and r['probe']==p and r['M']==m and abs(r['k']-1)<1e-6]
            Z[i,j]=v[0] if v else np.nan
    im=ax.imshow(Z,cmap='RdYlGn_r',vmin=0.8,vmax=3.2,aspect='auto')
    ax.set_xticks(range(len(Ms))); ax.set_xticklabels(Ms)
    ax.set_yticks(range(len(probes))); ax.set_yticklabels(probes)
    ax.set_xlabel('M'); ax.set_title(f'{dt}: main-task slowdown S_A (k=1)')
    for i in range(len(probes)):
        for j in range(len(Ms)):
            if not np.isnan(Z[i,j]): ax.text(j,i,f'{Z[i,j]:.2f}',ha='center',va='center',fontsize=8)
plt.tight_layout(); plt.savefig('figures/fig1_slowdown_heatmap.png',dpi=140); plt.close()
# fig2 overlap vs M
fig,ax=plt.subplots(figsize=(6.4,4))
for dt,ls in (('bf16','-'),('fp4','--')):
    for p,c in (('fp32','C0'),('dram','C1'),('tensor','C2')):
        ys=[100*np.mean([r['overlap'] for r in rows if r['dtype']==dt and r['probe']==p and r['M']==m]) for m in Ms]
        ax.plot(Ms,ys,ls,color=c,marker='o',label=f'{dt}/{p}')
ax.set_xscale('log',base=2); ax.set_xlabel('M'); ax.set_ylabel('overlap (%)')
ax.set_title('Concurrent overlap (wall-clock estimate)'); ax.grid(alpha=.3); ax.legend(fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig('figures/fig2_overlap.png',dpi=140); plt.close()
# fig3 pareto: main throughput vs probe throughput at M=128 and 1024
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,M in zip(axes,[128,1024]):
    for dt,mk in (('bf16','o'),('fp4','s')):
        for p,c in (('fp32','C0'),('dram','C1'),('tensor','C2')):
            xs=[];ys=[]
            for r in rows:
                if r['M']==M and r['dtype']==dt and r['probe']==p:
                    xs.append(1000*r['reps_A']/r['T_joint_ms']); ys.append(1000*r['reps_B']/r['T_joint_ms'])
            if xs: ax.scatter(xs,ys,marker=mk,color=c,alpha=.75,s=28,label=f'{dt}/{p}')
    ax.set_xlabel('main GEMM calls/s (in joint window)'); ax.set_ylabel('probe calls/s')
    ax.set_title(f'M={M}'); ax.grid(alpha=.3)
axes[0].legend(fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig('figures/fig3_pareto.png',dpi=140); plt.close()
json.dump(rows,open('raw/interference_full_with_overlap.json','w'),indent=2)
import statistics as st
print('rows',len(rows))
for p in probes:
    v=[r['overlap'] for r in rows if r['probe']==p]
    print('overlap %-6s mean=%.1f%% max=%.1f%%'%(p,100*st.mean(v),100*max(v)))
print('figs written')