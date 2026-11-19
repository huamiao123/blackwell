import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analyze import ROOT, read_run

class CounterIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        origin=next(p for p in sorted((ROOT/'raw').iterdir()) if (p/'case.json').is_file()
                    and json.loads((p/'case.json').read_text())['case'][2]=='resources')
        self.folder=Path(self.tmp.name)/origin.name
        shutil.copytree(origin,self.folder)
    def rewrite(self,key,value):
        file=self.folder/'counters.csv'
        with file.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
        rows[2][rows[0].index(key)]=value
        with file.open('w',newline='') as f:csv.writer(f).writerows(rows)
    def test_wrong_symbol(self):
        (self.folder/'profiled_symbols.json').write_text('["unrelated"]')
        with self.assertRaises(ValueError):read_run(self.folder)
    def test_wrong_geometry(self):
        self.rewrite('Grid Size','(1, 1, 1)')
        with self.assertRaises(ValueError):read_run(self.folder)
    def test_bad_hit_rate_preserves_raw_value(self):
        self.rewrite('lts__t_sector_hit_rate.pct','103.1')
        sample=next(r for r in read_run(self.folder) if r['metric']=='lts__t_sector_hit_rate.pct')
        self.assertEqual((sample['status'],sample['value']),('INVALID_HIT_PROBABILITY',103.1))
    def test_unavailable_is_not_zero(self):
        self.rewrite('dram__bytes_op_read.sum','n/a')
        with self.assertRaises(ValueError):read_run(self.folder)
