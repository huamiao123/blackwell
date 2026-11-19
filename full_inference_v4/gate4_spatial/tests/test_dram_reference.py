from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dram_reference import TILE_WORDS, output_word, read_digest, streaming_work


class DramReferenceTests(unittest.TestCase):
    def test_work_is_equal_across_grid_and_cta_shape(self):
        l2 = 96 * (1 << 20)
        for operation in ('read', 'write', 'copy'):
            light = streaming_work(operation, l2, 32, 'light')
            heavy = streaming_work(operation, l2, 170, 'heavy')
            for key in ('tiles', 'region_bytes', 'logical_read_bytes', 'logical_write_bytes'):
                self.assertEqual(light[key], heavy[key])
            self.assertEqual(light['region_bytes'], 4 * l2)
            self.assertEqual(heavy['physical_dram_bytes'], 'UNAVAILABLE')

    def test_staging_permutation_preserves_read_digest(self):
        tile = 9
        base = tile * TILE_WORDS
        words = [output_word(base + index, 142) for index in range(TILE_WORDS)]
        self.assertEqual(len(set(words)), TILE_WORDS)
        expected = sum(words[index ^ 32] * ((base + (index ^ 32)) | 1) for index in range(TILE_WORDS))
        self.assertEqual(read_digest(words, tile), expected & 0xffffffff)
