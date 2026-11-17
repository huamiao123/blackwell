"""CPU-only artifact hash inventory; exclude this inventory's own bytes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'analysis/gate2_artifact_manifest.json'


def main():
    if '--check' in sys.argv:
        d = json.loads(TARGET.read_text(encoding='utf-8'))
        for item in d['artifacts']:
            p = ROOT / item['path']
            assert p.stat().st_size == item['bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
        print('PASS:', len(d['artifacts']), 'published artifact hashes')
        return
    files = subprocess.check_output(['git', 'ls-files', '-c', '-o', '--exclude-standard', '-z', '--', ROOT.name], cwd=ROOT.parent)
    artifacts = []
    for name in sorted(set(n for n in files.decode('utf-8').split('\0') if n)):
        p = ROOT.parent / name
        if p == TARGET:
            continue
        artifacts.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    d = dict(scope='Gate-2 release; distinct from historical CPU preparation manifest',
             preparation_commit='979d88dd50b60a8ad9db192cff1300e51139b196',
             measured_source='accepted raw/*.json code_sha256 maps',
             exclusions='this manifest, ignored caches, large original traces/SQLite, disabled local GPU window',
             artifacts=artifacts)
    TARGET.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('INVENTORIED', len(artifacts), 'artifacts')


if __name__ == '__main__':
    main()
