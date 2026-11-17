"""Run preparation tests with GPU packages and model-weight reads denied."""
import importlib.abc,sys,unittest,builtins,io,os
from pathlib import Path

class DenyGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','cuda','cupy','triton','vllm'}:
            raise RuntimeError('GPU/model runtime import forbidden in CPU tests: '+fullname)
        return None

def safe_open(original):
    def wrapper(file,*args,**kwargs):
        if isinstance(file,(str,bytes,os.PathLike)):
            suffix=Path(os.fsdecode(file)).suffix.lower()
            if suffix in {'.safetensors','.pt','.pth','.bin'}:
                raise RuntimeError('model/weight-like file read forbidden in CPU tests')
        return original(file,*args,**kwargs)
    return wrapper

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--tests',required=True);a=p.parse_args()
    sys.meta_path.insert(0,DenyGpuImports());builtins.open=safe_open(builtins.open);io.open=safe_open(io.open)
    suite=unittest.defaultTestLoader.discover(a.tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
