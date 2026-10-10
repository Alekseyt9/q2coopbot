"""Replace only hash-bound copies in one sealed cohort with identical hardlinks."""
import argparse,collections,ctypes,json,os,pathlib,shutil,uuid
from ctypes import wintypes
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True); ap.add_argument('--apply',action='store_true'); a=ap.parse_args()
    root=a.root.resolve(); repo=pathlib.Path(__file__).resolve().parents[1]
    assert root.is_relative_to((repo/'workspace/artifacts').resolve()) and not a.out.exists()
    state=read(root/'progress.json'); assert state['stage']=='complete' and state['diagnostics_complete']
    assert state['quality_report_sha256']==sha(root/'quality-report.json')
    assert state['diagnostics_acceptance_sha256']==sha(root/'diagnostics-acceptance.json')
    proof=read(root/'recovery/verified-members.json'); protocol=read(root/'protocol.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(root/'protocol.json')
    # Actors running against this root are forbidden; read-only historical data is retained.
    command="$r=$env:Q2_COMBAT_MAINTENANCE_ROOT; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'q2ded.exe','q2coopbot.exe' -and $_.CommandLine -like ('*'+$r+'*') }).Count"
    import subprocess
    result=subprocess.check_output([str(repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'),'-NoProfile','-Command',command],
        env=dict(os.environ,Q2_COMBAT_MAINTENANCE_ROOT=str(root)),text=True)
    assert result.strip()=='0', 'Active actors reference this cohort'
    cache={}
    def digest(path):
        stat=path.stat(); key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache: cache[key]=sha(path)
        return cache[key]
    files={}
    for entry in protocol['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256']; plan=read(entry['plan'])
        for ti,task in enumerate(plan['tasks']):
            group=pathlib.Path(entry['root'])/f"case-{ti}-{task['modes'][0]}"
            for seed in task['seeds']:
                member=group/f's-{seed}'
                assert sha(member/'manifest.json')==proof['members'][str(member)]['manifest_sha256']
                manifest=read(member/'manifest.json')
                for name,field in [('q2coopbot.exe','client_sha256'),('q2combat-export.exe','exporter_sha256')]:
                    path=(member/name).resolve(); assert path.is_relative_to(root)
                    expected=manifest[field].lower(); assert digest(path)==expected
                    files[path]=expected
            aggregate=(group/'q2combat-export.exe').resolve()
            if aggregate.exists():
                assert aggregate.is_relative_to(root) and digest(aggregate)==manifest['exporter_sha256'].lower()
                files[aggregate]=manifest['exporter_sha256'].lower()
    before={p:(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_nlink) for p in files}
    groups=collections.defaultdict(list)
    for path,expected in files.items(): groups[(path.name,expected)].append(path)
    linked=0; already=0; errors=[]
    if a.apply:
        for paths in groups.values():
            candidates=[]
            for target in sorted(paths):
                target_inode=(target.stat().st_dev,target.stat().st_ino)
                if any((p.stat().st_dev,p.stat().st_ino)==target_inode for p in candidates): already+=1; continue
                source=next((p for p in candidates if p.stat().st_nlink<1000),None)
                if source is None: candidates.append(target); continue
                pending=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.link')
                assert target.resolve().is_relative_to(root) and pending.parent.resolve().is_relative_to(root) and not pending.exists()
                try:
                    os.link(source,pending)
                    os.replace(pending,target)
                    assert os.path.samefile(source,target) and digest(target)==files[target]
                    linked+=1
                except OSError as error:
                    errors.append(dict(path=str(target),error=str(error)))
                    if pending.exists(): pending.unlink()  # only this tool's explicitly checked temporary link
                    candidates.append(target)
    for path,expected in files.items(): assert digest(path)==expected
    after={p:(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_nlink) for p in files}
    before_unique={(v[0],v[1]):v[2] for v in before.values()}; after_unique={(v[0],v[1]):v[2] for v in after.values()}
    report=dict(state='complete' if not errors else 'complete_with_skips',applied=a.apply,paths=len(files),relinked=linked,already_shared=already,
        unique_inodes_before=len(before_unique),unique_inodes_after=len(after_unique),unique_logical_inode_bytes_before=sum(before_unique.values()),
        unique_logical_inode_bytes_after=sum(after_unique.values()),errors=errors,all_binary_sha256_preserved=True,
        protocol_sha256=sha(root/'protocol.json'),member_proof_sha256=sha(root/'recovery/verified-members.json'),
        scope='Only sealed hash-bound client/exporter copies and aggregate exporters. Every path and byte content retained, atomic replacements, hardlink groups capped below1000 links. Inode byte totals bound this root and are not exact whole-volume freed bytes; external hardlinks may retain allocation.')
    save(a.out,report); print(json.dumps({k:v for k,v in report.items() if k not in ('errors','scope')}))


if __name__=='__main__': main()
