"""CPU work accounting and deterministic references for the streaming probes."""

from math import ceil


TILE_WORDS = 8192
TILE_BYTES = TILE_WORDS * 4
WORD_MASK = (1 << 32) - 1


def output_word(index, seed):
    value = ((index ^ seed) * 0x9e3779b9) & WORD_MASK
    return value ^ (value >> 16)


def read_digest(words, tile):
    if len(words) != TILE_WORDS or tile < 0:
        raise ValueError('a complete input tile and nonnegative tile number are required')
    base = tile * TILE_WORDS
    return sum(int(word) * ((base + index) | 1) for index, word in enumerate(words)) & WORD_MASK


def streaming_work(operation, l2_bytes, grid, variant):
    if operation not in ('read', 'write', 'copy') or min(l2_bytes, grid) <= 0:
        raise ValueError('invalid streaming operation, cache size, or grid')
    threads, dynamic = {'light': (32, 0), 'heavy': (128, TILE_BYTES)}[variant]
    tiles = ceil(4 * l2_bytes / TILE_BYTES)
    if tiles * TILE_WORDS > 1 << 32:
        raise ValueError('region exceeds uint32 element indexing')
    region = tiles * TILE_BYTES
    reads = 0 if operation == 'write' else region
    writes = tiles * 4 if operation == 'read' else region
    return dict(operation=operation, variant=variant, tiles=tiles, region_bytes=region,
                logical_read_bytes=reads, logical_write_bytes=writes, grid=grid,
                nonempty_ctas=min(grid, tiles), threads_per_cta=threads,
                dynamic_shared_bytes=dynamic, physical_dram_bytes='UNAVAILABLE')
