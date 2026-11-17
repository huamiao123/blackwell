from pathlib import Path
import json,sqlite3,gzip,io,csv,shutil,hashlib
root=Path(__file__).resolve().parents[1]
for dbpath in sorted((root/'sqlite').glob('gate2_*.sqlite')):
 db=sqlite3.connect(dbpath);db.row_factory=sqlite3.Row
 tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 if 'GPU_METRICS' in tables:
  target=root/'analysis'/(dbpath.stem+'_metric_samples.csv.gz')
  if not target.exists():
   with target.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=5,mtime=0) as zipped,io.TextIOWrapper(zipped,encoding='utf-8',newline='') as text:
    w=csv.writer(text);w.writerow(['timestamp','typeId','metricId','metric','value'])
    for r in db.execute('SELECT g.timestamp,g.typeId,g.metricId,m.metricName,g.value FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m ON g.typeId=m.typeId AND g.metricId=m.metricId ORDER BY g.timestamp,g.typeId,g.metricId'):w.writerow(tuple(r))
  metadata=[dict(r) for r in db.execute('SELECT * FROM TARGET_INFO_GPU_METRICS')]
  (root/'analysis'/(dbpath.stem+'_metric_metadata.json')).write_text(json.dumps(metadata,indent=2))
 db.close()
for suffix in ['_activities.json','_kernel_catalog.csv']:
 for path in (root/'analysis').glob('gate2_*'+suffix):
  dest=path.with_name(path.name+'.gz')
  if not dest.exists() or dest.stat().st_mtime<path.stat().st_mtime:
   with path.open('rb') as src,dest.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=5,mtime=0) as out:shutil.copyfileobj(src,out)
manifest=[]
for folder,ext in [('raw','.nsys-rep'),('sqlite','.sqlite')]:
 for path in sorted((root/folder).glob('gate2_*'+ext)):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  manifest.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=h.hexdigest(),storage='remote original, not committed to Git'))
(root/'analysis/gate2_raw_trace_manifest.json').write_text(json.dumps(manifest,indent=2))
print('EXPORTED',len(manifest),'original trace/database hashes; lossless gzip data ready')
