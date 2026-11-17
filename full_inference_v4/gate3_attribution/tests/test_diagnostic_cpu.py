import datetime as dt
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from diagnostic_worker import sample_row, is_active_dispatch
from run_diagnostic import validate_window


class DiagnosticTests(unittest.TestCase):
    def test_final_empty_dispatch_is_not_a_model_sample(self):
        empty = dict(active_request_count=0, stage='EMPTY', scheduled_prefill_tokens=0, scheduled_decode_tokens=0)
        self.assertFalse(is_active_dispatch(empty))
        self.assertTrue(is_active_dispatch(dict(active_request_count=8, stage='DECODE')))
        for invalid in [None, dict(empty, scheduled_decode_tokens=1), dict(empty, stage='DECODE')]:
            with self.assertRaises(RuntimeError):
                is_active_dispatch(invalid)

    def test_association_requires_exact_requests(self):
        row = dict(requests=[dict(request_id='a', scheduled_tokens=1)])
        self.assertIs(sample_row([row], ['a']), row)
        for pending, ids in [([], ['a']), ([row, row], ['a']), ([row], ['b'])]:
            with self.assertRaises(RuntimeError):
                sample_row(pending, ids)

    def test_new_gate_never_reuses_old_window(self):
        now = dt.datetime.now(dt.timezone.utc)
        w = dict(enabled=True, gate=3, reviewed_gates=[0, 1, 2], allowed_configs=['E2_nvfp4'],
                 hostname='test', authorization_reference='test', not_before=now.isoformat(),
                 expires_at=(now + dt.timedelta(minutes=10)).isoformat())
        c = dict(experiment='E2', precision='nvfp4')
        self.assertGreater(validate_window(w, c, now.timestamp(), 'test'), now.timestamp())
        for change in [dict(gate=2), dict(enabled=False), dict(reviewed_gates=[0, 1]),
                       dict(expires_at=(now - dt.timedelta(seconds=1)).isoformat()),
                       dict(expires_at=(now + dt.timedelta(minutes=30)).isoformat())]:
            with self.assertRaises(ValueError):
                validate_window(dict(w, **change), c, now.timestamp(), 'test')


if __name__ == '__main__':
    unittest.main()
