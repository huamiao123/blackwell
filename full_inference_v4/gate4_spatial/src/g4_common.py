"""CPU-only shared file contracts for the three Gate-4 preparation tasks."""
from pathlib import Path
import csv,hashlib,json,math

V4=Path(__file__).resolve().parents[2]
SPATIAL=V4/'gate4_spatial'

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def vector(text):
    import ast
    x=ast.literal_eval(text)
    return x[0] if x and isinstance(x[0],(list,tuple)) else x
def product(xs):return math.prod(xs)
