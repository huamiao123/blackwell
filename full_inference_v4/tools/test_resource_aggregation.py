from pathlib import Path
import sys,ast,collections,statistics,math,random,json
base=Path(__file__).resolve().parents[1];sys.path.insert(0,str(base/'src'))
from analyze_resources import summarize_samples
source=(Path(__file__).parent/'validate_traces_remote.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fast_summarize')
ns=dict(collections=collections,statistics=statistics,math=math)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<fast summary>','exec'),ns)
fast=ns['fast_summarize'];rng=random.Random(4117)
for case in range(100):
 ss=[];t=0
 for i in range(20):
  t+=rng.choice([100,100,100,350])
  ss.append(dict(typeId=0,metricId=0,name='DRAM Read',timestamp=t,value=rng.random()*100))
 windows={'full':[(0,3000)],'pieces':[sorted([rng.randrange(3000),rng.randrange(3000)]) for _ in range(8)]}
 assert fast(ss,windows)==summarize_samples(ss,windows),case
out=base/'analysis/gate2_fast_resource_differential.json'
out.write_text(json.dumps(dict(cpu_only=True,random_cases=100,exact_match=True,
  purpose='One-time union of phase windows avoids repeated sorting; frozen worker sources unchanged'),indent=2))
print('100 differential cases exact; GPU worker sources unchanged')
