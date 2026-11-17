"""Guarded entry point: CPU fixtures are never rendered as experimental figures."""
from common import ROOT,load


if __name__=='__main__':
    actual=[p for p in (ROOT/'raw').glob('*.json') if load(p).get('schema')=='v4.1']
    if not actual:raise SystemExit('NO_VALID_GPU_DATA: figures pending Gate 2. No synthetic experimental figures generated.')
    raise SystemExit('Runtime data must pass timestamp/device/completeness audit before choosing figures; automatic plotting disabled.')
