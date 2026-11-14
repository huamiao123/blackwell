import json, statistics as st
rows=json.load(open('raw/interference_v2.json'))
def ov(tA,tB,tJ):
    hi,lo=max(tA,tB),min(tA,tB)
    return 0.0 if lo<=0 else max(0.0,min(1.0,1.0-(tJ-hi)/lo))
for r in rows:
    tA=r['T_A_ms']/r['S_A']; tB=r['T_B_ms']/r['S_B']
    r['alone_A_ms']=tA; r['alone_B_ms']=tB; r['overlap_est']=ov(tA,tB,r['T_joint_ms'])
print("M    dtype probe   k  aloneA aloneB joint  ovlp%")
for r in sorted(rows,key=lambda x:(x['M'],x['dtype'],x['probe'],x['k'])):
    if abs(r['k']-1.0)<1e-6:
        print("%5d %5s %6s %4.1f %7.2f %7.2f %6.2f %5.1f"%(r['M'],r['dtype'],r['probe'],r['k'],r['alone_A_ms'],r['alone_B_ms'],r['T_joint_ms'],r['overlap_est']*100))
print()
for p in ('fp32','dram','tensor'):
    v=[r['overlap_est'] for r in rows if r['probe']==p]
    print("mean overlap %-6s %.1f%%"%(p,100*st.mean(v)))
json.dump(rows, open('raw/interference_v2_with_overlap.json','w'), indent=2)