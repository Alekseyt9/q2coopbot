"""Discover physical nested execution plans without walking runtime aliases."""
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess


def discover_plan_files(root, exclude_roots=()):
    """Use the native file walker on Windows, without following runtime aliases."""
    root=Path(root).resolve()
    excluded=tuple(Path(p).resolve() for p in exclude_roots)
    if not root.exists() or any(root==p or root.is_relative_to(p) for p in excluded):return []
    executable=shutil.which('rg') if os.name=='nt' else None
    if executable:
        result=subprocess.run([executable,'--files','--hidden','--no-ignore','--iglob','*plan*.json','--',str(root)],
            capture_output=True,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode not in (0,1) or result.stderr.strip():
            raise RuntimeError('History file discovery failed: '+result.stderr)
        paths=[]
        for line in result.stdout.splitlines():
            path=Path(line)
            if any(path==p or path.is_relative_to(p) for p in excluded):continue
            # Match the Python walk's exclusion of file symlinks/reparse points.
            if path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&0x400:continue
            paths.append(path)
        return sorted(paths)
    return _discover_plan_files_walk(root,excluded)


def _discover_plan_files_walk(root, exclude_roots=()):
    root = Path(root).resolve()
    excluded = tuple(Path(p).resolve() for p in exclude_roots)
    pending = [root] if root.exists() and not any(root == p or root.is_relative_to(p) for p in excluded) else []
    found = []
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path)
                    if any(path == p or path.is_relative_to(p) for p in excluded):
                        continue
                    # NTFS junctions can point back into a runtime or source
                    # tree. Do not recurse through any directory reparse point.
                    attributes = getattr(entry.stat(follow_symlinks=False), 'st_file_attributes', 0)
                    if not entry.is_symlink() and not attributes & 0x400:
                        pending.append(path)
                elif fnmatch.fnmatch(entry.name, '*plan*.json') and entry.is_file(follow_symlinks=False):
                    found.append(Path(entry.path))
    return sorted(found)


def historical_plans(root, explicit=(), exclude_roots=()):
    """Yield unique (path, execution plan) pairs; reject unreadable history."""
    excluded = tuple(Path(p).resolve() for p in exclude_roots)
    paths = set(discover_plan_files(root, excluded))
    paths.update(Path(p).resolve() for p in explicit)
    for path in sorted(paths):
        if any(path == p or path.is_relative_to(p) for p in excluded):
            continue
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        # Some files named *plan*.json are arrays of paths or design notes.
        if not isinstance(value, dict) or not isinstance(value.get('tasks'), list):
            continue
        for task in value['tasks']:
            if not isinstance(task, dict) or not isinstance(task.get('seeds', []), list):
                raise ValueError('Invalid historical task: ' + str(path))
            if any(type(seed) is not int or seed < 0 for seed in task.get('seeds', [])):
                raise ValueError('Invalid historical seed: ' + str(path))
        yield path, value
