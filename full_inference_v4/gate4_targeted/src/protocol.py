"""Frozen positions for an interleaved P2 diagnostic and full-request P0 blocks."""
import random

MODES=('original','sham','o_parallel','down_parallel','o_ab','down_ab','o_ba','down_ba')
STEPS=tuple(range(33,65))

def balanced(seed, blocks):
    rng=random.Random(seed)
    result=[]
    for _ in range(blocks):
        group=list(MODES)
        rng.shuffle(group)
        result.extend(group)
    return result

def schedules(profile, seed):
    if profile=='P2':return [balanced(seed,4)]
    return [[mode]*len(STEPS) for mode in balanced(seed,5)]
