"""Inspect installed vLLM and TorchInductor outputs on the experiment host (CPU only)."""
import ast
import hashlib
import json
from pathlib import Path

root=Path.home()/'fp4-llm'
source=root/'vllm/vllm/model_executor/models/qwen3.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3Attention')
forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
needle='triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_'
seen={}
for p in sorted((root/'cache/vllm').rglob('*.py')):
    b=p.read_bytes()
    if needle.encode() not in b:continue
    sha=hashlib.sha256(b).hexdigest()
    if sha in seen:continue
    module=ast.parse(b)
    kernels=[]
    for item in ast.walk(module):
        if not isinstance(item,ast.Constant) or not isinstance(item.value,str) or 'def '+needle not in item.value:continue
        try:sub=ast.parse(item.value)
        except SyntaxError:continue
        for fn in sub.body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith(needle):continue
            ops={}
            for call in ast.walk(fn):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr in ('load','store','atomic_add','atomic_cas'):
                    ops.setdefault(call.func.attr,set()).add(ast.unparse(call.args[0]).split(' ')[0])
            footprint={k:sorted(v) for k,v in ops.items()}
            if footprint!={'load':['in_ptr0','in_ptr1','in_ptr2','in_ptr3'],'store':['out_ptr1']}:
                raise ValueError('unrecognized memory traffic '+str(footprint))
            kernels.append(dict(name=fn.name,memory=footprint,source_sha256=hashlib.sha256(ast.unparse(fn).encode()).hexdigest()))
    if kernels:seen[sha]=dict(path=str(p.relative_to(root)),kernels=kernels)
if not seen:raise ValueError('compiled wrapper evidence unavailable')
print(json.dumps(dict(model_source=str(source.relative_to(root)),model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      forward_lines=[forward.lineno,forward.end_lineno],compiled_sources=seen),indent=2))
