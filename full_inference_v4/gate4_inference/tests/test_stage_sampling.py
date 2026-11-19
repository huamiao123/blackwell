from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_gate44_neighbors import union_ns
from analyze_gate44_stage_metrics import weighted


class StageSamplingTests(unittest.TestCase):
    def test_overlapping_primary_intervals_are_counted_once(self):
        self.assertEqual(union_ns([(120, 150), (100, 140), (160, 175)]), 65)

    def test_sampling_clips_both_window_edges_and_ignores_gaps_between_replays(self):
        r = weighted([(0, 10), (100, 30), (200, 50), (300, 0)],
                     [(50, 120), (250, 290)], 100)
        self.assertEqual(r['window_ns'], 110)
        self.assertEqual(r['interval_coverage'], 1)
        self.assertAlmostEqual(r['mean_pct'], (50*10+20*30+40*50)/110)

    def test_large_sampling_gap_invalidates_short_window(self):
        with self.assertRaises(ValueError):
            weighted([(0, 30), (100, 30), (500, 90)], [(0, 400)], 100)
