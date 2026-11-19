"""Measurement regressions checked against the archived real runs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import analyze

class TimingTests(unittest.TestCase):
    def test_partial_overlap_and_gap(self):
        samples=[]
        for step,mode,qs,qe,ks,ke in [(1,'original',10,30,35,60),(2,'sham',10,30,35,60),(3,'fork',10,30,15,40)]:
            samples.append(dict(mode=mode,step=step,graph_ns=2000,
                           layers=[[i,5,qs,qe,ks,ke,90,95,1,2] for i in range(36)]))
        _,rows=analyze.timeline({'replays':samples},{})
        rows={r['mode']:r for r in rows}
        self.assertAlmostEqual(rows['original']['sum_gap_and_hideable_median_us'],36*.025)
        self.assertEqual(rows['sham']['overlap_hits'],0)
        self.assertEqual(rows['fork']['overlap_hits'],36)
        self.assertAlmostEqual(rows['fork']['pair_us_median'],.030)

    def test_real_nodes_and_requests(self):
        old=analyze.OUT
        with TemporaryDirectory() as directory:
            try:
                analyze.OUT=Path(directory)
                analyze.analyze()
                rows=lambda name: list(analyze.csv.DictReader((analyze.OUT/name).read_text().splitlines()))
                p0=rows('p0_trials.csv')
                self.assertEqual(len(p0),60)
                self.assertEqual(len(rows('p2_steps.csv')),1024)
                self.assertEqual(len(rows('whole_request_scale.csv')),4)
                self.assertEqual(sum(r['status']=='COMPLETED' for r in rows('supervision.csv')),8)
                self.assertEqual(set((r['precision'],r['rep']) for r in p0),{(p,r) for p in ('bf16','nvfp4') for r in ('r1','r2')})
            finally:
                analyze.OUT=old
