"""Read-only migration plan: AAS plus their retained immutable backing files."""
import os, json, hashlib, argparse
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',required=True); p.add_argument('--out',required=True)
a=p.parse_args(); root=Path(a.root).resolve(); store=root/'build/aas-store'
by_inode={}; groups={}; candidates=[]
for directory, dirs, files in os.walk(root):
    dirs[:]=[d for d in dirs if not Path(directory,d).is_junction() and not Path(directory,d).is_symlink() and Path(directory,d)!=store]
    for name in files:
        path=Path(directory,name)
        if path.is_symlink(): continue
        if path.suffix.lower()=='.aas':
            s=path.stat(); key=(s.st_dev,s.st_ino)
            if key not in by_inode:
                with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
                by_inode[key]=digest
            digest=by_inode[key]
            g=groups.setdefault(digest,{'sha256':digest,'bytes':s.st_size,'source':str(path),'paths':[]})
            g['paths'].append(str(path))
        elif 'runtime-immutable' in path.parts or 'runtime-paks' in path.parts:
            candidates.append(path)
sizes={g['bytes'] for g in groups.values()}
for path in candidates:
    s=path.stat()
    if s.st_size not in sizes: continue
    key=(s.st_dev,s.st_ino)
    digest=by_inode.get(key)
    if digest is None:
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest in groups: groups[digest]['paths'].append(str(path))
report={'root':str(root),'store':str(store),'groups':list(groups.values()),'paths':sum(len(g['paths']) for g in groups.values()),'content_bytes':sum(g['bytes'] for g in groups.values())}
Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
