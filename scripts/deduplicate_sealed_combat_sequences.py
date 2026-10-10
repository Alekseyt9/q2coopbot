"""Keep one immutable inode for byte-identical native/final CUDA sequence copies."""
import argparse
import ctypes
import os
import pathlib
import subprocess
import uuid
from ctypes import wintypes

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    boundary = (repo / 'workspace/artifacts').resolve()
    assert not args.out.exists()
    pwsh = repo / 'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    allocated = kernel.GetCompressedFileSizeW
    allocated.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    allocated.restype = wintypes.DWORD
    def physical(path):
        high = wintypes.DWORD()
        low = allocated(str(path), ctypes.byref(high))
        assert low != 0xffffffff or ctypes.get_last_error() == 0
        return (high.value << 32) + low
    records = []
    for root in args.roots:
        root = root.resolve()
        assert root.is_relative_to(boundary)
        assert read(root / 'progress.json')['stage'] == 'complete'
        assert read(root / 'report.json')['state'] == 'complete'
        command = r"$r=$env:Q2_SEQUENCE_MAINTENANCE_ROOT; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python.*|q2ppo-data)\.exe$' -and $_.CommandLine -and ($_.CommandLine.Contains('--out '+$r) -or $_.CommandLine.Contains('--out '+[char]34+$r)) }).Count"
        result = subprocess.check_output([str(pwsh), '-NoProfile', '-Command', command],
            env=dict(os.environ, Q2_SEQUENCE_MAINTENANCE_ROOT=str(root)), text=True)
        assert result.strip() == '0', 'A producer still targets this closed root'
        for native in root.glob('*/rollout-*-native'):
            final = native.with_name(native.name.removesuffix('-native'))
            source, target = (native / 'sequence.jsonl').resolve(), (final / 'sequence.jsonl').resolve()
            assert source.is_relative_to(root) and target.is_relative_to(root)
            if not source.exists() or not target.exists():
                continue
            raw, verified = read(native / 'report.json'), read(final / 'report.json')
            assert verified['numerical_verification'] == 'cuda_verified_v1'
            assert raw['sequence_sha256'] == verified['sequence_sha256']
            expected = raw['sequence_sha256']
            assert sha(source) == sha(target) == expected
            shared = os.path.samefile(source, target)
            reclaimed_bound = 0
            if not shared:
                assert source.stat().st_nlink < 1000
                target_links = target.stat().st_nlink
                reclaimed_bound = physical(target) if target_links == 1 else 0
                pending = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.link')
                assert pending.parent.resolve().is_relative_to(root) and not pending.exists()
                try:
                    os.link(source, pending)
                    os.replace(pending, target)
                finally:
                    if pending.exists():
                        pending.unlink()
                assert os.path.samefile(source, target) and sha(target) == expected
            records.append(dict(source=str(source), target=str(target), sha256=expected, already_shared=shared,
                logical_bytes=source.stat().st_size, unique_target_allocation_before=reclaimed_bound))
    save(args.out, dict(state='complete', pairs=records, relinked=sum(not r['already_shared'] for r in records),
        unique_target_allocation_bound=sum(r['unique_target_allocation_before'] for r in records),
        scope='Complete processing roots only; native/final CUDA sequence SHA exact. Both paths and all bytes retained. '
              'Atomic identical hardlinks; external links excluded from reclaimed allocation bound. No models, checkpoints or training outcomes removed.'))
    print(dict(pairs=len(records), relinked=sum(not r['already_shared'] for r in records),
               allocation_bound=sum(r['unique_target_allocation_before'] for r in records)), flush=True)


if __name__ == '__main__':
    main()
