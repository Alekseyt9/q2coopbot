"""Resume interrupted paired native evaluation without replaying valid battles.

Uses the existing per-episode PowerShell harness, independent 16-slot queue and
strict member verifier. Does not train or perform model numerical checks.
"""
import argparse, concurrent.futures, datetime, hashlib, json, msvcrt, pathlib, subprocess, sys, time
from process_combat_architecture_pool import read, sha, save, run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--port',type=int,default=34700)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1];root=a.root.resolve()
    assert root.is_relative_to(repo/'workspace/artifacts')
    assert 1024<=a.port and a.port+15<=65535
    pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    protocol=read(root/'protocol.json');assert protocol['slots']==16 and protocol['timescale']==2
    # Only the recovery driver holds this OS lock; stale files do not keep it locked.
    recovery=root/'recovery';recovery.mkdir(exist_ok=True)
    lock=(recovery/'resume.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    command=(f"$ports=@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object {{$_.LocalPort -ge {a.port} -and $_.LocalPort -lt {a.port+16}}}); "
             "$live=@(Get-CimInstance Win32_Process | Where-Object {($_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_combat_checkpoint_series.py') -or ($_.Name -eq 'pwsh.exe' -and $_.CommandLine -match '\\s-File\\s+.*run_combat_architecture_evaluation.ps1')}); "
             "@{ports=$ports.Count;drivers=@($live | ForEach-Object {$_.ProcessId})} | ConvertTo-Json -Compress")
    safety=read_json_command([pwsh,'-NoProfile','-Command',command])
    assert safety['ports']==0 and not safety['drivers'],'Existing evaluation driver or occupied pool ports'
    jobs=[];reference=None;cache={}
    def digest(path):
        path=pathlib.Path(path);stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:
            with path.open('rb') as f:cache[key]=hashlib.file_digest(f,'sha256').hexdigest()
        return cache[key]
    for pi,entry in enumerate(protocol['evaluations']):
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan'])
        assert sha(plan['registry_path'])==plan['registry_sha256']
        if plan.get('model_path'):assert sha(plan['model_path'])==entry['deterministic_weights_sha256']==plan['model_sha256']
        for ti,task in enumerate(plan['tasks']):
            assert sha(task['runner_path'])==task['runner_sha256']
            assert sha(repo/task['episode']['recipe']['reward_config'])==task['reward_sha256']
            assert len(task['modes'])==1
            for si,seed in enumerate(task['seeds']):
                mode=task['modes'][0];folder=pathlib.Path(entry['root'])/f'case-{ti}-{mode}'/f's-{seed}'
                assert folder.resolve().is_relative_to(root)
                jobs.append(dict(job=len(jobs),plan_index=pi,task_index=ti,seed_index=si,seed=seed,mode=mode,
                                 root=str(folder),plan=entry['plan'],entry=entry,task=task))
    assert len(jobs)==protocol['total_episodes']
    existing={read(p)['job']:read(p) for p in (root/'pool/jobs').glob('*-result.json')}
    assert set(existing)<=set(range(len(jobs)))
    def validate(job):
        nonlocal reference
        folder=pathlib.Path(job['root']);r=read(folder/'report.json');m=read(folder/'manifest.json')
        assert r['capture_complete'] and r['provenance_valid'] and len(r['results'])==1
        row=r['results'][0];assert row['seed']==job['seed']
        assert all(row[k] for k in ('capture_valid','dispatch_valid','seed_confirmed','frame_budget_valid'))
        task=job['task'];expected=job['entry']['deterministic_weights_sha256'] or ''
        assert (m.get('model_weights_sha256') or '').lower()==expected
        assert m['reward_config_sha256'].lower()==task['reward_sha256']
        if task.get('instances'):
            assert row['generated_fixture']==task['instances'][job['seed_index']] and row['generated_start']['confirmed']
        if reference is None:
            for item in m['sources']:assert digest(repo/item['path'])==item['sha256']
            for item in m['native_sources']:assert digest(repo.parent/'yquake2'/item['path'])==item['sha256']
            reference=m
        else:
            assert m['sources']==reference['sources'] and m['native_sources']==reference['native_sources']
            assert m['source_fingerprint']==reference['source_fingerprint']
            assert m['native_source_fingerprint']==reference['native_source_fingerprint']
        for name,field in [('q2combat-export.exe','exporter_sha256'),('q2coopbot.exe','client_sha256')]:
            assert digest(folder/name)==m[field].lower()
        for item in row['runtime_files']:
            assert digest(pathlib.Path(row['root'])/'runtime'/item['path'])==item['sha256'].lower()
    completed=[];pending=[];salvaged=[]
    for job in jobs:
        if job['job'] in existing:
            old=existing[job['job']]
            assert not old['error']
            assert all(old[k]==job[k] for k in ('plan_index','task_index','seed','mode','root'))
            validate(job);completed.append(job)
        elif (pathlib.Path(job['root'])/'report.json').exists():
            # A terminal native report can survive interruption before its queue receipt.
            validate(job);completed.append(job);salvaged.append(job)
        else:pending.append(job)
    assert reference is not None,'Resume requires at least one verified native member'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt=dict(version='combat_evaluation_resume_v1',state='verified',total=len(jobs),completed_reused=len(completed),
                 queued=len(pending),salvaged=[j['job'] for j in salvaged],protocol_sha256=sha(root/'protocol.json'),
                 source_fingerprint=reference['source_fingerprint'],native_source_fingerprint=reference['native_source_fingerprint'],
                 slots=16,timescale=2,primary_receipts_preserved=True)
    print(json.dumps(receipt),flush=True)
    if a.dry_run:return
    save(recovery/f'resume-{stamp}.json',receipt)
    for job in salvaged:
        save(root/f"pool/jobs/job-{job['job']}-result.json",dict(
            **{k:job[k] for k in ('job','plan_index','task_index','seed','mode','root')},
            slot=None,port=None,started_utc=None,finished_utc=None,wall_seconds=None,error='',
            recovered_from_terminal_native_report=True))
    # Preserve interrupted, unsealed members outside their original expected paths.
    for job in pending:
        folder=pathlib.Path(job['root']).resolve()
        if folder.exists():
            target=(recovery/'interrupted'/stamp/f"job-{job['job']}").resolve()
            assert folder.is_relative_to(root) and target.is_relative_to(root) and folder!=root
            target.parent.mkdir(parents=True,exist_ok=True);folder.rename(target)
    pending.sort(key=lambda j:(j['seed_index'],j['task_index'],j['plan_index'],j['mode']))
    queue=__import__('queue').Queue()
    for job in pending:queue.put(job)
    save(recovery/'resume-progress.json',dict(state='capturing',completed=len(completed),total=len(jobs),queued=len(pending)))
    def worker(slot):
        outputs=[]
        while True:
            try:job=queue.get_nowait()
            except __import__('queue').Empty:break
            start=datetime.datetime.now(datetime.timezone.utc).isoformat();timer=time.perf_counter();error=''
            try:
                run([pwsh,'-NoProfile','-File',repo/'scripts/run_registered_combat_pool_episode.ps1',
                     '-Plan',job['plan'],'-TaskIndex',job['task_index'],'-SeedIndex',job['seed_index'],
                     '-Mode',job['mode'],'-Port',a.port+slot,'-OutputRoot',job['root']],
                    root/f"pool/jobs/job-{job['job']}-resume-{stamp}.log")
            except Exception as ex:error=str(ex)
            result=dict(**{k:job[k] for k in ('job','plan_index','task_index','seed','mode','root')},
                        slot=slot,port=a.port+slot,started_utc=start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        wall_seconds=time.perf_counter()-timer,error=error,resume_run=stamp)
            save(root/f"pool/jobs/job-{job['job']}-result.json",result);outputs.append(result)
            if error:break
        return outputs
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        outputs=[r for batch in executor.map(worker,range(16)) for r in batch]
    assert len(outputs)==len(pending) and not any(j['error'] for j in outputs),'Resumed capture failed; retained receipts'
    run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',root,
         '--source-fingerprint',reference['source_fingerprint'],'--native-fingerprint',reference['native_source_fingerprint']],
        recovery/f'verify-{stamp}.log')
    proof=recovery/'verified-members.json'
    run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',root,'--member-proof',proof],
        recovery/f'quality-{stamp}.log')
    quality=read(root/'quality-report.json');assert quality['state']=='complete' and quality['episodes']==len(jobs)
    save(recovery/'resume-progress.json',dict(state='complete',completed=len(jobs),total=len(jobs),
         quality_report_sha256=sha(root/'quality-report.json')))
    print(json.dumps(dict(state='complete',episodes=len(jobs),quality=str(root/'quality-report.md'))),flush=True)


def read_json_command(command):
    result=subprocess.run([str(x) for x in command],capture_output=True,text=True,check=True,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    return json.loads(result.stdout)


if __name__=='__main__':main()
