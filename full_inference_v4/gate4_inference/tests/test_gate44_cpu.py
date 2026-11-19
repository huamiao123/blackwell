from copy import deepcopy
import datetime as dt
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_gate44_p0 import request_itl
from gate44_worker import PAIRED_SEQUENCES, targets
from run_gate44 import validate_window


class Gate44Tests(unittest.TestCase):
    def test_probes_are_fixed_and_conditions_paired(self):
        self.assertEqual(targets('prefill'), (0,))
        self.assertEqual(len(targets('decode')), 32)
        self.assertEqual(targets('lifecycle'), (0, *targets('decode')))
        for block in PAIRED_SEQUENCES:
            self.assertEqual(sorted(block), ['a_alone', 'empty', 'parallel', 'serial'])

    def test_scope_cannot_be_expanded_with_the_same_window(self):
        now = dt.datetime(2026, 11, 19, 12, tzinfo=dt.timezone.utc)
        window = dict(enabled=True, gate='4.4', hostname='host', authorization_reference='user-authorized trial',
                      allowed_workload=dict(experiment='E2', precision='nvfp4', profile='P2',
                                            phase='decode', condition='parallel'),
                      not_before=now.isoformat(), expires_at=(now + dt.timedelta(minutes=5)).isoformat())
        config = dict(experiment='E2', precision='nvfp4')
        validate_window(window, config, now.timestamp(), 'host', 'decode', 'parallel', 'P2')
        for phase, condition, profile in [('prefill', 'parallel', 'P2'), ('decode', 'serial', 'P2'),
                                          ('decode', 'parallel', 'P0')]:
            with self.assertRaises(ValueError):
                validate_window(window, config, now.timestamp(), 'host', phase, condition, profile)

    def test_request_itl_refuses_bundled_outputs(self):
        request = {'output_events': [dict(count=1, time_ns=100 + i * 1000000) for i in range(257)]}
        self.assertEqual(request_itl(request), [1.0] * 256)
        bundled = deepcopy(request)
        bundled['output_events'][0]['count'] = 2
        with self.assertRaises(ValueError):
            request_itl(bundled)
