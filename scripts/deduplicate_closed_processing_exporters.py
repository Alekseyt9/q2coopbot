"""Share hash-bound exporter copies in completed processing roots only."""
import argparse
import os
import pathlib
import subprocess
import uuid

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    boundary = (repo / 'workspace/artifacts').resolve()
    assert not args.out.exists()
    candidates = []
    receipts = []
    for root in args.roots:
        root = root.resolve()
        assert root.is_relative_to(boundary)
        assert read(root / 'progress.json')['stage'] == 'complete'
        assert read(root / 'report.json')['state'] == 'complete'
        command = r"$r=$env:Q2_PROCESSING_DEDUP_ROOT; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python.*|q2ppo-data|q2combat-export)\.exe$' -and $_.CommandLine -and $_.CommandLine.Contains($r) }).Count"
        count = subprocess.check_output([
            str(repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'),
            '-NoProfile', '-Command', command], text=True,
            env=dict(os.environ, Q2_PROCESSING_DEDUP_ROOT=str(root)))
        assert count.strip() == '0', 'Processing producer/reader still references this root'
        receipts.append(dict(root=str(root), report_sha256=sha(root / 'report.json')))
        for path in root.glob('*/case-*/q2combat-export.exe'):
            assert path.resolve().is_relative_to(root)
            manifest = path.parent / 'manifest.json'
            expected = read(manifest)['exporter_sha256'].lower()
            assert sha(path) == expected
            candidates.append((path, expected, sha(manifest)))
    # Finish validation of every input before any replacement.
    before = {p.stat().st_ino: p.stat().st_size for p, _, _ in candidates}
    sources = {}
    records = []
    for target, expected, manifest_sha in candidates:
        group = sources.setdefault(expected, [])
        source = next((p for p in group if os.path.samefile(p, target)), None)
        shared = source is not None
        if source is None:
            source = next((p for p in group if p.stat().st_nlink < 1000), None)
            if source is None:
                group.append(target)
            else:
                pending = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.link')
                assert pending.parent.resolve().is_relative_to(boundary) and not pending.exists()
                try:
                    os.link(source, pending)
                    os.replace(pending, target)
                finally:
                    if pending.exists():
                        pending.unlink()
                assert os.path.samefile(source, target)
        assert sha(target) == expected and sha(target.parent / 'manifest.json') == manifest_sha
        records.append(dict(path=str(target), sha256=expected, manifest_sha256=manifest_sha,
                            relinked=source is not None and not shared))
    after = {p.stat().st_ino: p.stat().st_size for p, _, _ in candidates}
    report = dict(state='complete', roots=receipts, records=records,
                  relinked=sum(r['relinked'] for r in records),
                  unique_inodes_before=len(before), unique_inodes_after=len(after),
                  unique_inode_bytes_before=sum(before.values()), unique_inode_bytes_after=sum(after.values()),
                  scope='Only completed processing case exporters bound by manifest SHA. All paths, bytes and manifests retained. No model or training data changes; inode totals are not whole-volume freed-byte measurements.')
    save(args.out, report)
    print({k: v for k, v in report.items() if k not in ('roots', 'records', 'scope')}, flush=True)


if __name__ == '__main__':
    main()
