"""Independent interval oracle: protects physical-overlap measurement semantics."""
import random
import unittest
from analyze_overlap import union, intersection, duration
from overlap_runner import prompt_ids


class MeasurementChecks(unittest.TestCase):
    def test_concurrent_kernels_are_not_summed(self):
        self.assertEqual(duration([(0, 10), (5, 15)]), 15)
        self.assertEqual(duration(intersection([(0, 10), (5, 15)], [(8, 20)])), 7)
        self.assertEqual(intersection([(0, 10)], [(10, 15)]), [])

    def test_random_intervals_against_discrete_occupancy_oracle(self):
        rng = random.Random(5711)
        for _ in range(200):
            a = [sorted([rng.randrange(30), rng.randrange(30)]) for _ in range(12)]
            b = [sorted([rng.randrange(30), rng.randrange(30)]) for _ in range(12)]
            occupied_a = {i for s, e in a for i in range(s, e)}
            occupied_b = {i for s, e in b for i in range(s, e)}
            self.assertEqual(duration(a), len(occupied_a))
            self.assertEqual(duration(intersection(a, b)), len(occupied_a & occupied_b))

    def test_prompts_reproducible_distinct_and_b1_matches_b8(self):
        ids = prompt_ids(8)
        self.assertEqual(len(set(map(tuple, ids))), 8)
        self.assertEqual(prompt_ids(1)[0], ids[0])
        self.assertTrue(all(len(row) == 2048 for row in ids))
        self.assertTrue(all(0 <= token < 151936 for row in ids for token in row))


if __name__ == '__main__':
    unittest.main()
