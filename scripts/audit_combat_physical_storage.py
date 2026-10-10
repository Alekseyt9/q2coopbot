"""Bounded snapshot of already sealed compression receipts and binary inodes."""
import argparse, ctypes, hashlib, json, pathlib, sys
from ctypes import wintypes
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True); a=ap.parse_args()
    assert sys.platform=='win32' and not a.out.exists()
    root=a.root.resolve(); receipt_path=root/'stream-compression-receipts.json'
    data=receipt_path.read_bytes(); receipt=json.loads(data)
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCompressedFileSizeW.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
    api.GetCompressedFileSizeW.restype=wintypes.DWORD
    def physical(path):
        high=wintypes.DWORD(); ctypes.set_last_error(0)
        low=api.GetCompressedFileSizeW(str(path),ctypes.byref(high))
        if low==0xffffffff and ctypes.get_last_error(): raise ctypes.WinError(ctypes.get_last_error())
        return (high.value<<32)|low
    files=[]
    for record in receipt['files']:
        path=pathlib.Path(record['path']).resolve(); assert path.is_relative_to(root)
        assert sha(path)==record['sha256'] and path.stat().st_size==record['logical_bytes']
        files.append(dict(path=str(path),sha256=record['sha256'],logical_bytes=path.stat().st_size,physical_bytes=physical(path)))
    binaries={}
    for name in ('q2coopbot.exe','q2combat-export.exe'):
        paths=[]
        for jobpath in (root/'pool/jobs').glob('*-result.json'):
            try: job=read(jobpath)
            except (OSError,ValueError): continue
            if not job['error']: paths.append(pathlib.Path(job['root'])/name)
        unique={(p.stat().st_dev,p.stat().st_ino):p for p in paths}
        binaries[name]=dict(completed_member_paths=len(paths),unique_inodes=len(unique),
            logical_bytes=sum(p.stat().st_size for p in paths),unique_physical_bytes=sum(physical(p) for p in unique.values()))
    save(a.out,dict(state='complete_snapshot',compression_receipts_snapshot_sha256=hashlib.sha256(data).hexdigest(),
        snapshot_completed_members=receipt['completed_members'],streams=len(files),
        logical_stream_bytes=sum(f['logical_bytes'] for f in files),physical_stream_bytes=sum(f['physical_bytes'] for f in files),
        files=files,binaries=binaries,
        scope='Only streams in this atomic receipt snapshot; all SHA256 verified. NTFS physical sizes and binary inode counts measured. Not whole-root or whole-disk usage, not final battle quality.'))
    print(json.dumps(dict(streams=len(files),logical_bytes=sum(f['logical_bytes'] for f in files),physical_bytes=sum(f['physical_bytes'] for f in files),binaries=binaries)))


if __name__=='__main__': main()
