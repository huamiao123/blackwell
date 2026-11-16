"""CPU-only verification of the committed measured artifacts and provenance."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json

def sha(path,compressed=False):
    h=hashlib.sha256();size=0
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
    return size,h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent.parent)
    root=ap.parse_args().data_root.resolve()
    manifest=json.loads((root/'publication_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path=(root/entry['path']).resolve()
        assert path.is_relative_to(root),entry['path']
        assert sha(path)==(entry['bytes'],entry['sha256']),entry['path']
    for entry in manifest['compressed']:
        assert sha(root/entry['path'],compressed=True)==(entry['uncompressed_bytes'],entry['uncompressed_sha256']),entry['path']
    rows=list(csv.DictReader((root/'baseline.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==40
    configs={(int(r['batch']),r['precision'],r['mode']) for r in rows}
    assert configs=={(b,p,m) for b in [1,8] for p in ['bf16','nvfp4'] for m in ['eager','graph']}
    for cfg in configs:
        subset=[r for r in rows if (int(r['batch']),r['precision'],r['mode'])==cfg]
        assert len(subset)==5
        assert all(int(r['decode_steps'])==256 and int(r['actual_batch'])==cfg[0] for r in subset)
        assert all(int(r['graph_replays'])==(256 if cfg[2]=='graph' else 0) for r in subset)
    with gzip.open(root/'analysis/report_data.json.gz','rt',encoding='utf-8') as f:bundle=json.load(f)
    for name,expected in bundle['script_sha256'].items():
        assert sha(root/'src/original'/name)[1]==expected,name
    assert not json.loads((root/'analysis/quality_audit.json').read_text())['issues']
    print(f"PASS: {len(manifest['files'])} source artifacts, lossless compressed data, 40 baselines / 8 configurations, and original script hashes")

if __name__=='__main__':main()
