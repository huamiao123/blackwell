from pathlib import Path
import sys,datetime as dt,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_bf16 import validate_window

class ScopeTests(unittest.TestCase):
    def test_bf16_scope_and_expiry(self):
        now=dt.datetime.now(dt.timezone.utc)
        w=dict(enabled=True,gate=3,reviewed_gates=[0,1,2],allowed_configs=['E2_bf16'],hostname='test',
               authorization_reference='test',not_before=now.isoformat(),expires_at=(now+dt.timedelta(minutes=10)).isoformat())
        c=dict(experiment='E2',precision='bf16')
        self.assertGreater(validate_window(w,c,now.timestamp(),'test'),now.timestamp())
        for bad in [dict(precision='nvfp4'),dict(experiment='E1')]:
            with self.assertRaises(ValueError):validate_window(w,dict(c,**bad),now.timestamp(),'test')
        for bad in [dict(enabled=False),dict(expires_at=(now-dt.timedelta(seconds=1)).isoformat()),dict(allowed_configs=['E2_nvfp4'])]:
            with self.assertRaises(ValueError):validate_window(dict(w,**bad),c,now.timestamp(),'test')

if __name__=='__main__':unittest.main()
