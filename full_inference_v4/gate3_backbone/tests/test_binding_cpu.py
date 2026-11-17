from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graph_binding import match_references


class BindingTests(unittest.TestCase):
    def test_unique_parameter_owner(self):
        t = [dict(module='layer0.qkv_proj', tensor='alpha', address=0x123456789000)]
        b = b'abcd' + t[0]['address'].to_bytes(8, 'little') + b'x'*16
        r = match_references(b, t)
        self.assertEqual(r['module'], 'layer0.qkv_proj')
        self.assertEqual(r['references'][0]['byte_offset'], 4)

    def test_ambiguity_never_resolved_by_order(self):
        ts = [dict(module='a', tensor='weight', address=4096), dict(module='b', tensor='alpha', address=8192)]
        r = match_references((4096).to_bytes(8,'little')+(8192).to_bytes(8,'little'), ts)
        self.assertEqual(r['status'], 'AMBIGUOUS_MODULE_REFERENCES')
        self.assertIsNone(r['module'])

    def test_empty_zero_and_partial_pointer_are_not_matches(self):
        ts = [dict(module='a', tensor='alpha', address=0),dict(module='b', tensor='weight', address=0x123456789000)]
        self.assertEqual(match_references(b'\0'*32,ts)['status'], 'NO_EXACT_REFERENCE')
        self.assertEqual(match_references((0x123456789000).to_bytes(8,'little')[:7],ts)['status'], 'NO_EXACT_REFERENCE')


if __name__ == '__main__':
    unittest.main()
