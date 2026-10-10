[CmdletBinding()]
param([Parameter(Mandatory)][string[]]$Plans,
 [ValidateSet(4,8,16,24,32)][int]$MaxInstances=16,[int]$Port=34800,
 [Parameter(Mandatory)][string]$OutputRoot,[switch]$DryRun,[switch]$NoBinaryBundle)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path $repo $OutputRoot}))
if(Test-Path -LiteralPath $root){throw 'Fresh pool output required'}
if($Port -lt 1024 -or $Port+$MaxInstances-1 -gt 65535){throw 'Invalid port range'}
$schedules=@();$outputs=@{}
foreach($path in $Plans){
    $full=(Resolve-Path -LiteralPath $path).Path
    & "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan $full -DryRun
    $s=Get-Content -LiteralPath $full -Raw|ConvertFrom-Json
    if(@($s.tasks|Where-Object {$_.episode.recipe.runner -ne 'combat-baseline'}).Count){throw 'Episode pool requires combat-baseline recipes; use -Scheduler cohort for full campaign runner'}
    $target=[IO.Path]::GetFullPath($s.output_root)
    if($outputs.ContainsKey($target) -or (Test-Path -LiteralPath $target) -or $target -eq $root){throw 'Duplicate or occupied plan output'}
    $outputs[$target]=$true
    $schedules+=@{path=$full;sha256=(Get-FileHash $full).Hash.ToLowerInvariant();schedule=$s}
}
if(!$schedules.Count){throw 'No plans'}
$jobs=[Collections.Generic.List[object]]::new();$groups=@();$index=0;$ordinal=0
for($p=0;$p -lt $schedules.Count;$p++){
    $s=$schedules[$p].schedule
    for($t=0;$t -lt $s.tasks.Count;$t++){
        $task=$s.tasks[$t]
        foreach($mode in $task.modes){
            $groupRoot=Join-Path $s.output_root "case-$t-$mode"
            $group=@{plan_index=$p;task_index=$t;mode=$mode;root=$groupRoot;task=$task;ordinal=$ordinal}
            $groups+=,$group
            for($i=0;$i -lt $task.seeds.Count;$i++){
                $jobs.Add(@{index=$index;plan_index=$p;task_index=$t;seed_index=$i;seed=$task.seeds[$i];mode=$mode;plan=$schedules[$p].path;root=(Join-Path $groupRoot "s-$($task.seeds[$i])");ordinal=$ordinal})
                $index++
            }
            $ordinal++
        }
    }
}
if($DryRun){Write-Output "Verified episode pool: $MaxInstances independent slots, $($jobs.Count) queued battles";return}
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $Port -and $_.LocalPort -lt $Port+$MaxInstances}).Count){throw 'Pool port range occupied'}
New-Item -ItemType Directory -Path $root|Out-Null
New-Item -ItemType Directory -Path "$root/jobs"|Out-Null
foreach($s in $schedules){New-Item -ItemType Directory -Path $s.schedule.output_root|Out-Null;Copy-Item -LiteralPath $s.path -Destination (Join-Path $s.schedule.output_root 'plan.json')}
foreach($g in $groups){New-Item -ItemType Directory -Path $g.root|Out-Null}
. "$PSScriptRoot/harness_manifest.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
. "$PSScriptRoot/combat_harness_bundle.ps1"
$binaryBundle=''
if(!$NoBinaryBundle){$binaryBundle=New-HarnessBinaryBundle $repo}
$queue=[Collections.Concurrent.ConcurrentQueue[object]]::new()
foreach($job in @($jobs|Sort-Object seed_index,task_index,plan_index,mode)){$queue.Enqueue($job)}
$hostExe=(Get-Process -Id $PID).Path;$monitor=$null;$clock=[Diagnostics.Stopwatch]::StartNew()
try{
    $monitor=Start-Process $hostExe -ArgumentList @('-NoProfile','-File',"`"$PSScriptRoot/monitor_combat_pool.ps1`"",'-Port',$Port,'-Count',$MaxInstances,'-Output',"`"$root/resources.jsonl`"",'-StopFile',"`"$root/monitor.stop`"") -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/monitor.stdout" -RedirectStandardError "$root/monitor.stderr"
    $results=@(0..($MaxInstances-1)|ForEach-Object -Parallel {
        $ErrorActionPreference='Stop';$slot=$_;$workQueue=$using:queue;$job=$null
        while($workQueue.TryDequeue([ref]$job)){
            $start=[DateTime]::UtcNow;$timer=[Diagnostics.Stopwatch]::StartNew();$failure=''
            @{state='running';job=$job.index;seed=$job.seed;mode=$job.mode;port=$using:Port+$slot}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
            try{
                & "$using:repo/scripts/run_registered_combat_pool_episode.ps1" -Plan $job.plan -TaskIndex $job.task_index -SeedIndex $job.seed_index -Mode $job.mode -Port ($using:Port+$slot) -OutputRoot $job.root -BinaryBundle $using:binaryBundle *>&1|Set-Content "$using:root/jobs/job-$($job.index).log"
            }catch{$failure=$_.Exception.Message}
            $result=[pscustomobject]@{job=$job.index;plan_index=$job.plan_index;task_index=$job.task_index;seed=$job.seed;mode=$job.mode;slot=$slot;port=$using:Port+$slot;root=$job.root;started_utc=$start.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');wall_seconds=$timer.Elapsed.TotalSeconds;error=$failure}
            $result|ConvertTo-Json|Set-Content "$using:root/jobs/job-$($job.index)-result.json"
            $result
            $job=$null
        }
        @{state='idle'}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
    } -ThrottleLimit $MaxInstances)
    $unchanged=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $records=@();$failures=@($results|Where-Object error);$frames=0;$episodes=0
    foreach($g in $groups){
        $members=@($results|Where-Object {$_.plan_index -eq $g.plan_index -and $_.task_index -eq $g.task_index -and $_.mode -eq $g.mode}|Sort-Object seed)
        if($members.Count -ne $g.task.seeds.Count -or @($members|Where-Object error).Count){continue}
        $all=@();$receipts=@();$manifest=$null;$groupExporter=''
        foreach($m in $members){
            $r=Get-Content "$($m.root)/report.json" -Raw|ConvertFrom-Json
            $mf=Get-Content "$($m.root)/manifest.json" -Raw|ConvertFrom-Json
            if(!$r.capture_complete -or !$r.provenance_valid -or $r.results.Count -ne 1 -or $r.results[0].seed -ne $m.seed -or !$r.results[0].capture_valid -or !$r.results[0].dispatch_valid -or !$r.results[0].seed_confirmed){throw 'Native member changed'}
            $memberExporter=Join-Path $m.root 'q2combat-export.exe'
            if((Get-FileHash -LiteralPath $memberExporter).Hash.ToLowerInvariant() -ne $mf.exporter_sha256.ToLowerInvariant()){throw 'Native member exporter changed'}
            if(!$manifest){$manifest=$mf;$groupExporter=$memberExporter}
            elseif($manifest.source_fingerprint -ne $mf.source_fingerprint -or $manifest.native_source_fingerprint -ne $mf.native_source_fingerprint -or $manifest.model_weights_sha256 -ne $mf.model_weights_sha256 -or $manifest.reward_config_sha256 -ne $mf.reward_config_sha256 -or $manifest.exporter_sha256 -ne $mf.exporter_sha256){throw 'Group member provenance differs'}
            $all+=@($r.results)
            $receipts+=@{root=$m.root;seed=$m.seed;report_sha256=(Get-FileHash "$($m.root)/report.json").Hash.ToLowerInvariant();manifest_sha256=(Get-FileHash "$($m.root)/manifest.json").Hash.ToLowerInvariant()}
        }
        Copy-Item -LiteralPath $groupExporter -Destination (Join-Path $g.root 'q2combat-export.exe')
        if((Get-FileHash -LiteralPath (Join-Path $g.root 'q2combat-export.exe')).Hash.ToLowerInvariant() -ne $manifest.exporter_sha256.ToLowerInvariant()){throw 'Aggregate exporter copy changed'}
        $manifest|Add-Member -NotePropertyName pool_members -NotePropertyValue $receipts
        $manifest.workers=1;$manifest.episodes_per_worker=$all.Count;$manifest.seeds=@($all.seed);$manifest.seed_assignments=@($all|Select-Object worker,episode,seed,port,fixture_mixed,solo_fixture)
        if($g.task.instances.Count){@{instances=$g.task.instances}|ConvertTo-Json -Depth 18|Set-Content "$($g.root)/generated-fixtures.json" -Encoding utf8NoBOM;$manifest|Add-Member -NotePropertyName generated_fixtures -NotePropertyValue "$($g.root)/generated-fixtures.json" -Force;$manifest.generated_fixtures_sha256=(Get-FileHash "$($g.root)/generated-fixtures.json").Hash.ToLowerInvariant()}
        $manifest|ConvertTo-Json -Depth 25|Set-Content "$($g.root)/manifest.json" -Encoding utf8NoBOM
        @{version=1;provenance_valid=$unchanged;capture_complete=$unchanged;timescale=2;parallelism=$MaxInstances;scheduler='independent_episode_queue';wall_seconds=$clock.Elapsed.TotalSeconds;completed_episodes=$all.Count;usable_captures=$all.Count;results=$all;pool_members=$receipts}|ConvertTo-Json -Depth 25|Set-Content "$($g.root)/report.json" -Encoding utf8NoBOM
        $s=$schedules[$g.plan_index].schedule
        $record=@{episode_id=$g.task.episode.id;episode_revision=$g.task.episode.revision;episode_sha256=$g.task.episode_sha256;split=$g.task.split;mode=$g.mode;seeds=@($all.seed);artifacts=$g.root;report_sha256=(Get-FileHash "$($g.root)/report.json").Hash.ToLowerInvariant();model_sha256=$(if($g.mode -eq 'learned'){$s.model_sha256}else{''});capture_verified=$unchanged}
        $record|ConvertTo-Json -Depth 10|Set-Content "$($g.root)/registry-binding.json" -Encoding utf8NoBOM
        $records+=@{plan_index=$g.plan_index;record=$record}
        $episodes+=$all.Count;$frames+=($all|Measure-Object actual_game_frames -Sum).Sum
    }
    $valid=$unchanged -and !$failures.Count -and $results.Count -eq $jobs.Count -and $records.Count -eq $groups.Count
    for($p=0;$p -lt $schedules.Count;$p++){
        $same=$schedules[$p].sha256 -eq (Get-FileHash $schedules[$p].path).Hash.ToLowerInvariant();$valid=$valid -and $same
        @{state=$(if($valid -and $same){'complete'}else{'failed'});source_unchanged=$unchanged;pool_root=$root;records=@($records|Where-Object plan_index -eq $p|ForEach-Object {$_.record})}|ConvertTo-Json -Depth 20|Set-Content (Join-Path $schedules[$p].schedule.output_root 'report.json') -Encoding utf8NoBOM
    }
    @{state=$(if($valid){'complete'}else{'failed'});max_instances=$MaxInstances;cohort_workers=1;slots=$MaxInstances;scheduler='independent_episode_queue';wall_seconds=$clock.Elapsed.TotalSeconds;usable_captures=$episodes;actual_game_frames=$frames;source_unchanged=$unchanged;jobs=$results;plans=@($schedules|ForEach-Object {@{path=$_.path;sha256=$_.sha256;model_sha256=$_.schedule.model_sha256}})}|ConvertTo-Json -Depth 22|Set-Content "$root/report.json" -Encoding utf8NoBOM
    if(!$valid){throw 'Episode pool incomplete; inspect job results'}
    Write-Output $root
}finally{Set-Content "$root/monitor.stop" 'complete' -Encoding ascii;if($monitor -and !$monitor.WaitForExit(10000)){throw 'Monitor did not finish'}}
