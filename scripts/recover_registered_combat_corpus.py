"""Recover interrupted native collection without changing frozen source/plans.

Keep independently closed members, quarantine incomplete members, rerun missing
jobs with the existing episode runner, reuse the original pool finalizer, then
perform the ordinary CUDA replay. Requires the pilot OS lock and free UDP ports.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from process_combat_architecture_pool import read,sha
from run_registered_combat_ppo_pilot import lock,save


PS_PREFIX=r'''
$ErrorActionPreference='Stop'
$ctx=Get-Content -LiteralPath $args[0] -Raw|ConvertFrom-Json
$repo=$ctx.repo;$root=$ctx.pool;$Port=[int]$ctx.port;$MaxInstances=16
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $Port -and $_.LocalPort -lt $Port+$MaxInstances}).Count){throw 'Recovery port range occupied'}
. "$repo/scripts/harness_manifest.ps1"
. "$repo/scripts/combat_harness_bundle.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
if($fingerprint -ne $ctx.source_fingerprint){throw 'Recovery source differs from retained captures'}
$nativeFingerprint=Get-HarnessFingerprint (Get-HarnessNativeSourceRecords $ctx.native_repo)
if($nativeFingerprint -ne $ctx.native_source_fingerprint){throw 'Recovery native source differs'}
$schedules=@()
foreach($p in $ctx.plans){
 if((Get-FileHash -LiteralPath $p.path).Hash.ToLowerInvariant() -ne $p.sha256){throw 'Frozen plan changed'}
 & "$repo/scripts/run_registered_combat_episodes.ps1" -Plan $p.path -DryRun
 $schedules+=@{path=$p.path;sha256=$p.sha256;schedule=(Get-Content -LiteralPath $p.path -Raw|ConvertFrom-Json)}
}
$groups=@($ctx.groups);$jobs=@($ctx.jobs)
$binaryBundle=New-HarnessBinaryBundle $repo
$queue=[Collections.Concurrent.ConcurrentQueue[object]]::new()
foreach($job in @($jobs|Where-Object {$_.index -in $ctx.missing}|Sort-Object seed_index,task_index,plan_index,mode)){$queue.Enqueue($job)}
$kept=@($ctx.kept);$hostExe=(Get-Process -Id $PID).Path;$monitor=$null;$clock=[Diagnostics.Stopwatch]::StartNew()
try{
 $monitor=Start-Process $hostExe -ArgumentList @('-NoProfile','-File',"`"$repo/scripts/monitor_combat_pool.ps1`"",'-Port',$Port,'-Count',16,'-Output',"`"$($ctx.recovery)/resources.jsonl`"",'-StopFile',"`"$($ctx.recovery)/monitor.stop`"") -WindowStyle Hidden -PassThru -RedirectStandardOutput "$($ctx.recovery)/monitor.stdout" -RedirectStandardError "$($ctx.recovery)/monitor.stderr"
 $fresh=@(0..15|ForEach-Object -Parallel {
  $ErrorActionPreference='Stop';$slot=$_;$workQueue=$using:queue;$job=$null
  while($workQueue.TryDequeue([ref]$job)){
   $start=[DateTime]::UtcNow;$timer=[Diagnostics.Stopwatch]::StartNew();$failure=''
   @{state='running';job=$job.index;seed=$job.seed;mode=$job.mode;port=$using:Port+$slot}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
   try{
    & "$using:repo/scripts/run_registered_combat_pool_episode.ps1" -Plan $job.plan -TaskIndex $job.task_index -SeedIndex $job.seed_index -Mode $job.mode -Port ($using:Port+$slot) -OutputRoot $job.root -BinaryBundle $using:binaryBundle *>&1|Set-Content "$using:root/jobs/job-$($job.index).log"
   }catch{$failure=$_.Exception.Message}
   $result=[pscustomobject]@{job=$job.index;plan_index=$job.plan_index;task_index=$job.task_index;seed=$job.seed;mode=$job.mode;slot=$slot;port=$using:Port+$slot;root=$job.root;started_utc=$start.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$timer.Elapsed.TotalSeconds;error=$failure;recovery_rerun=$true}
   $result|ConvertTo-Json|Set-Content "$using:root/jobs/job-$($job.index)-result.json"
   $result;$job=$null
  }
  @{state='idle'}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
 } -ThrottleLimit 16)
 $results=@($kept)+@($fresh)
 foreach($m in $results){
  if($m.error){throw "Recovery member failed: $($m.job)"}
  $manifest=Get-Content "$($m.root)/manifest.json" -Raw|ConvertFrom-Json
  if($manifest.source_fingerprint -ne $ctx.source_fingerprint -or $manifest.native_source_fingerprint -ne $ctx.native_source_fingerprint){throw 'Cross-session member source differs'}
 }
'''
PS_SUFFIX=r'''
 $report=Get-Content "$root/report.json" -Raw|ConvertFrom-Json
 $report|Add-Member -NotePropertyName recovery -NotePropertyValue @{receipt=$ctx.receipt;retained=$kept.Count;rerun=$fresh.Count;timing_scope='wall_seconds measures recovery session only; original member timings remain separate'}
 $report|ConvertTo-Json -Depth 25|Set-Content "$root/report.json" -Encoding utf8NoBOM
}finally{Set-Content "$($ctx.recovery)/monitor.stop" 'complete' -Encoding ascii;if($monitor -and !$monitor.WaitForExit(10000)){throw 'Recovery monitor did not finish'}}
'''


def main():
    ap=argparse.ArgumentParser()
    for name in ('corpus','pilot'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--port',type=int,required=True);args=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];corpus=args.corpus.resolve();pilot=args.pilot.resolve()
    assert corpus.is_relative_to(repo/'workspace/artifacts') and pilot.is_relative_to(repo/'workspace/artifacts')
    assert 1024<=args.port<=65520
    assert not (corpus/'complete.json').exists() and not (corpus/'pool/report.json').exists()
    assert shutil.disk_usage(corpus.anchor).free>=32*1024**3
    with lock(pilot/'execution.lock'):
        # No source edits or model/objective substitutions during recovery.
        for item in read(pilot/'execution-source.json')['files']:assert sha(item['path'])==item['sha256']
        protocol=read(corpus/'protocol.json');assert protocol['train_only'] and protocol['ppo_training_objective']
        for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
        subprocess.run([shutil.which('pwsh'),'-NoProfile','-Command',
            f"if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {{$_.LocalPort -ge {args.port} -and $_.LocalPort -lt {args.port+16}}}).Count){{throw 'Recovery port range occupied'}}"],
            check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        expected_model=sha(pathlib.Path(protocol['source'])/'weights.json')
        recovery=corpus/'recovery-v1';assert not recovery.exists();recovery.mkdir()
        plans=[];jobs=[];groups=[]
        for pi,assignment in enumerate(protocol['assignments']):
            plan=pathlib.Path(assignment['plan']);assert sha(plan)==assignment['plan_sha256'];schedule=read(plan)
            plans.append(dict(path=str(plan),sha256=sha(plan)))
            for ti,task in enumerate(schedule['tasks']):
                for mode in task['modes']:
                    group=pathlib.Path(schedule['output_root'])/f'case-{ti}-{mode}'
                    assert group.resolve().is_relative_to(corpus)
                    groups.append(dict(plan_index=pi,task_index=ti,mode=mode,root=str(group),task=task,ordinal=len(groups)))
                    for si,seed in enumerate(task['seeds']):
                        jobs.append(dict(index=len(jobs),plan_index=pi,task_index=ti,seed_index=si,seed=seed,mode=mode,plan=str(plan),root=str(group/f's-{seed}')))
        kept=[];missing=[];pins={};base=None
        for job in jobs:
            member=pathlib.Path(job['root']);report_path=member/'report.json';valid=False
            if report_path.exists():
                report=read(report_path);manifest=read(member/'manifest.json');values=report.get('results',[])
                valid=bool(report.get('capture_complete') and report.get('provenance_valid') and len(values)==1 and values[0]['seed']==job['seed'] and all(values[0].get(k) for k in ('capture_valid','dispatch_valid','seed_confirmed','frame_budget_valid')))
                if valid:
                    assert manifest['model_weights_sha256'].lower()==expected_model
                    assert manifest['reward_config_sha256'].lower()==protocol['ppo_training_objective']['reward_sha256']
                    for key in ('source_fingerprint','native_source_fingerprint','model_weights_sha256','reward_config_sha256','exporter_sha256'):
                        if base is not None:assert str(manifest[key]).lower()==str(base[key]).lower(),key
                    base=base or manifest
                    assert sha(member/'q2combat-export.exe')==manifest['exporter_sha256'].lower()
                    paths=[report_path,member/'manifest.json',member/'q2combat-export.exe']
                    paths.extend(pathlib.Path(values[0]['root'])/n for n in ('server.log','bot.jsonl'))
                    for p in paths:pins[str(p)]=sha(p)
                    result_path=corpus/'pool/jobs'/f"job-{job['index']}-result.json"
                    if result_path.exists():
                        result=read(result_path);assert not result['error']
                        for key in ('plan_index','task_index','seed','mode','root'):assert result[key]==job[key]
                    else:
                        result=dict(job=job['index'],**{k:job[k] for k in ('plan_index','task_index','seed','mode','root')},
                            slot=None,port=values[0]['port'],started_utc=None,finished_utc=None,wall_seconds=report.get('wall_seconds'),error='',
                            recovered_closed_capture=True,timing_scope='capture report wall duration; scheduler timestamps unavailable')
                        save(result_path,result)
                    kept.append(result)
            if not valid:missing.append(job['index'])
        assert base and kept and len(kept)+len(missing)==len(jobs)
        save(recovery/'retained-inputs.json',dict(state='complete',inputs=[dict(path=p,sha256=h) for p,h in pins.items()]))
        archived=[]
        for job in jobs:
            if job['index'] not in missing:continue
            member=pathlib.Path(job['root']);destination=recovery/'interrupted-members'/f"job-{job['index']}"
            destination.mkdir(parents=True)
            for original,name in [(member,'capture'),(pathlib.Path(str(member)+'-fixture.json'),'fixture.json'),
                    (corpus/'pool/jobs'/f"job-{job['index']}-result.json",'job-result.json'),
                    (corpus/'pool/jobs'/f"job-{job['index']}.log",'job.log')]:
                target=destination/name
                assert original.resolve().is_relative_to(corpus) and target.resolve().is_relative_to(recovery)
                if original.exists():original.rename(target);archived.append(dict(source=str(original),archived=str(target)))
        receipt=recovery/'native-recovery.json'
        context=dict(repo=str(repo),native_repo=str(repo.parent/'yquake2'),pool=str(corpus/'pool'),port=args.port,
            recovery=str(recovery),receipt=str(receipt),plans=plans,jobs=jobs,groups=groups,kept=kept,missing=missing,
            source_fingerprint=base['source_fingerprint'],native_source_fingerprint=base['native_source_fingerprint'])
        save(recovery/'context.json',context)
        template=repo/'scripts/run_registered_combat_episode_pool.ps1';text=template.read_text(encoding='utf-8-sig')
        start=text.index('    $unchanged=$fingerprint');end=text.index('}finally{Set-Content',start)
        script=recovery/'resume-native.ps1';script.write_text(PS_PREFIX+text[start:end]+PS_SUFFIX,encoding='utf-8-sig')
        save(recovery/'preparation.json',dict(state='complete',retained=len(kept),rerun=len(missing),archived=archived,
            original_finalizer_sha256=sha(template),recovery_script_sha256=sha(script),script_sha256=sha(__file__),training_performed=False))
        def run(label,command):
            with (recovery/(label+'.log')).open('w',encoding='utf-8') as output:
                p=subprocess.run(list(map(str,command)),cwd=repo,stdout=output,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            assert p.returncode==0,'Inspect '+str(recovery/(label+'.log'))
        save(recovery/'progress.json',dict(stage='recovering-native',retained=len(kept),rerun=len(missing)))
        run('native',[shutil.which('pwsh'),'-NoProfile','-File',script,recovery/'context.json'])
        pool=read(corpus/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==len(jobs)
        for p,h in pins.items():assert sha(p)==h,'Retained capture changed: '+p
        save(receipt,dict(state='complete',retained=len(kept),rerun=len(missing),pool_sha256=sha(corpus/'pool/report.json'),
            preparation_sha256=sha(recovery/'preparation.json'),training_performed=False))
        processed=[]
        for index,assignment in enumerate(protocol['assignments']):
            label=assignment['label'];target=corpus/('processing-'+label);assert not target.exists()
            save(recovery/'progress.json',dict(stage='cuda-replay',label=label))
            run('cuda-'+label,[sys.executable,repo/'scripts/process_registered_combat_pool_cuda.py','--pool',corpus/'pool','--plan-index',index,'--out',target])
            report=read(target/'report.json');assert report['state']=='complete' and report['device']=='cuda' and not report['training_performed']
            assert len(report['cases'])==len(assignment['seeds'])
            processed.append(dict(label=label,cases=len(report['cases']),rows=report['rows'],report=str(target/'report.json'),sha256=sha(target/'report.json')))
        for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
        save(corpus/'complete.json',dict(state='complete',version='combat_aim_feedback_corpus_v1',protocol_sha256=sha(corpus/'protocol.json'),
            pool_sha256=sha(corpus/'pool/report.json'),processed=processed,native_cases=len(jobs),cuda_cases=sum(p['cases'] for p in processed),
            training_performed=False,promotion=False,script_sha256=sha(__file__),native_recovery=str(receipt),native_recovery_sha256=sha(receipt)))
        save(corpus/'progress.json',dict(stage='complete',processed=processed,recovered=True))
        save(recovery/'progress.json',dict(stage='complete',retained=len(kept),rerun=len(missing)))
        print(json.dumps(dict(state='complete',retained=len(kept),rerun=len(missing),cases=len(jobs))))


if __name__=='__main__':main()
