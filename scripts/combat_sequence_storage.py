"""Share immutable sequence bytes, with bounded streaming across volumes."""
import hashlib,os,pathlib,shutil


def preserve_sequence(source,target,digest):
    source,target=pathlib.Path(source),pathlib.Path(target)
    assert source.is_file() and not target.exists()
    def sha(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    assert sha(source)==digest
    try:
        os.link(source,target)
    except OSError:
        # An existing target is never silently replaced. Avoid CopyFile2,
        # which failed on a compressed sequence with WinError1450.
        if target.exists():raise
        with source.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst,length=1024*1024)
        method='stream-copy'
    else:method='hardlink'
    assert source.stat().st_size==target.stat().st_size and sha(source)==sha(target)==digest
    return method
