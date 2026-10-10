"""Retain historical executable paths and bytes while sharing identical NTFS files."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    boundary = (repo / 'workspace/artifacts').resolve()
    out = args.out.resolve()
    assert out.is_relative_to(boundary) and not out.exists()
    pwsh = repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    command = "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(q2.*|yquake2|quake2|python.*|go)\\.exe$' } | Select-Object ExecutablePath,CommandLine) | ConvertTo-Json -Compress"
    processes = json.loads(subprocess.check_output([str(pwsh), '-NoProfile', '-Command', command], text=True) or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    jobs = {}
    roots = []
    for record in json.loads(args.inventory.read_text()):
        root = pathlib.Path(record['root']).resolve()
        assert root.is_relative_to(boundary)
        if root == out or any(str(root).lower() in str(p).lower() for p in processes):
            continue
        if 'demo' in root.name or root.is_junction() or root.is_symlink():
            continue
        roots.append(str(root))
        for directory, dirs, names in os.walk(root):
            dirs[:] = [n for n in dirs if not os.lstat(pathlib.Path(directory) / n).st_file_attributes & 1024]
            for name in names:
                if not name.lower().endswith('.exe'):
                    continue
                path = pathlib.Path(directory) / name
                stat = path.stat(follow_symlinks=False)
                if stat.st_file_attributes & 1024 or stat.st_size < 65536:
                    continue
                assert path.resolve().is_relative_to(root)
                key = (stat.st_dev, stat.st_ino)
                item = jobs.setdefault(key, dict(path=path, paths=[], size=stat.st_size, mtime=stat.st_mtime_ns))
                item['paths'].append(path)
    out.mkdir()
    def save(name, value):
        target = out / name
        pending = target.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2))
        os.replace(pending, target)
    free_before = shutil.disk_usage(out).free
    save('inventory.json', dict(roots=roots, unique_executables=len(jobs), paths=sum(len(j['paths']) for j in jobs.values()),
                               unique_logical_bytes=sum(j['size'] for j in jobs.values())))
    def verify(job):
        value = digest(job['path'])
        stat = job['path'].stat()
        assert (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (job['path'].stat().st_ino, job['size'], job['mtime'])
        return job, value
    anchors = collections.defaultdict(list)
    replaced = 0
    started = time.time()
    with (out / 'replacements.jsonl').open('w') as receipt, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, (job, value) in enumerate(pool.map(verify, jobs.values()), 1):
            group = anchors[(job['size'], value)]
            for target in job['paths']:
                before = target.stat()
                assert before.st_size == job['size'] and before.st_mtime_ns == job['mtime']
                if any(os.path.samefile(target, p) for p in group):
                    continue
                source = next((p for p in group if p.stat().st_nlink < 950), None)
                if source is None:
                    group.append(target)
                    continue
                assert digest(target) == value
                pending = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.link')
                assert pending.parent.resolve().is_relative_to(boundary) and not pending.exists()
                try:
                    os.link(source, pending)
                    assert target.stat().st_ino == before.st_ino
                    os.replace(pending, target)
                finally:
                    if pending.exists():
                        pending.unlink()
                assert os.path.samefile(source, target) and digest(target) == value
                receipt.write(json.dumps(dict(path=str(target), source=str(source), sha256=value, bytes=before.st_size)) + '\n')
                receipt.flush()
                replaced += 1
            if index % 100 == 0:
                save('progress.json', dict(stage='deduplicating', processed=index, total=len(jobs), replaced=replaced,
                                          free_bytes=shutil.disk_usage(out).free))
                print(dict(processed=index, total=len(jobs), replaced=replaced), flush=True)
    save('report.json', dict(state='complete', roots=roots, replaced=replaced, free_before=free_before,
                             free_after=shutil.disk_usage(out).free, seconds=time.time() - started,
                             scope='Inactive historical EXE copies only. Every path retained; SHA before/after exact; groups capped below950 links. Models, checkpoints and traces unchanged. Volume free delta also includes concurrent activity.'))
    save('progress.json', dict(stage='complete', replaced=replaced))
    print(dict(state='complete', replaced=replaced, free_bytes=shutil.disk_usage(out).free), flush=True)


if __name__ == '__main__':
    main()
