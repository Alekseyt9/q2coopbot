[CmdletBinding()]
param([Parameter(Mandatory)][string[]]$Plans,
 [ValidateSet(4,8,16,24,32)][int]$MaxInstances=16,
 [ValidateRange(1024,65500)][int]$Port=34800,
 [Parameter(Mandatory)][string]$OutputRoot,
 [ValidateSet('episode','cohort')][string]$Scheduler='episode',[switch]$DryRun)
$ErrorActionPreference='Stop'
if($Scheduler -eq 'episode'){
    & "$PSScriptRoot/run_registered_combat_episode_pool.ps1" -Plans $Plans -MaxInstances $MaxInstances -Port $Port -OutputRoot $OutputRoot -DryRun:$DryRun
    return
}
$repo=Split-Path $PSScriptRoot -Parent
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path $repo $OutputRoot}))
if(Test-Path -LiteralPath $root){throw 'Fresh pool output required'}
if($Port+$MaxInstances-1 -gt 65535){throw 'Pool port range invalid'}
$schedules=@();$outputs=@{}
foreach($path in $Plans){
    $full=(Resolve-Path -LiteralPath $path).Path
    & "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan $full -DryRun
    $schedule=Get-Content -LiteralPath $full -Raw|ConvertFrom-Json
    $target=[IO.Path]::GetFullPath($schedule.output_root)
    if($outputs.ContainsKey($target) -or (Test-Path -LiteralPath $target) -or $target -eq $root){throw 'Duplicate or occupied plan output'}
    $outputs[$target]=$true
    $schedules+=@{path=$full;sha256=(Get-FileHash $full).Hash.ToLowerInvariant();schedule=$schedule}
}
if(!$schedules.Count){throw 'No plans'}
$slots=[int]($MaxInstances/4)
if($DryRun){Write-Output "Verified pool: $MaxInstances instances, $slots concurrent case/model cohorts";return}
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $Port -and $_.LocalPort -lt $Port+$MaxInstances}).Count){throw 'Pool port range occupied'}
New-Item -ItemType Directory -Path $root|Out-Null
New-Item -ItemType Directory -Path "$root/jobs"|Out-Null
. "$PSScriptRoot/harness_manifest.ps1"
$sourceFingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$queue=[Collections.Concurrent.ConcurrentQueue[object]]::new()
$pendingJobs=[Collections.Generic.List[object]]::new()
$jobIndex=0
for($p=0;$p -lt $schedules.Count;$p++){
    $schedule=$schedules[$p].schedule
    New-Item -ItemType Directory -Path $schedule.output_root|Out-Null
    Copy-Item -LiteralPath $schedules[$p].path -Destination (Join-Path $schedule.output_root 'plan.json')
    $ordinal=0
    foreach($task in $schedule.tasks){foreach($mode in $task.modes){
        # Keep the existing four-instance cohort verifier. Cohorts from
        # different cases/models occupy disjoint reusable four-port slots.
        $sub=($schedule|ConvertTo-Json -Depth 30)|ConvertFrom-Json
        $one=($task|ConvertTo-Json -Depth 30)|ConvertFrom-Json
        $one.modes=@($mode);$sub.tasks=@($one)
        $sub.output_root=Join-Path $schedule.output_root "pool-job-$jobIndex"
        $planFile="$root/jobs/job-$jobIndex-plan.json"
        $sub|ConvertTo-Json -Depth 30|Set-Content -LiteralPath $planFile -Encoding utf8NoBOM
        $pendingJobs.Add(@{index=$jobIndex;plan_index=$p;ordinal=$ordinal;plan=$planFile;root=$sub.output_root;episode=$one.episode.id;mode=$mode})
        $ordinal++
        $jobIndex++
    }}
}
foreach($job in @($pendingJobs|Sort-Object ordinal,plan_index)){$queue.Enqueue($job)}
$hostExe=(Get-Process -Id $PID).Path
$monitor=$null
$clock=[Diagnostics.Stopwatch]::StartNew()
try{
    $monitor=Start-Process $hostExe -ArgumentList @('-NoProfile','-File',"`"$PSScriptRoot/monitor_combat_pool.ps1`"",'-Port',$Port,'-Count',$MaxInstances,'-Output',"`"$root/resources.jsonl`"",'-StopFile',"`"$root/monitor.stop`"") -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/monitor.stdout" -RedirectStandardError "$root/monitor.stderr"
    $results=@(0..($slots-1)|ForEach-Object -Parallel {
        $ErrorActionPreference='Stop';$slot=$_;$workQueue=$using:queue
        $job=$null
        while($workQueue.TryDequeue([ref]$job)){
            $timer=[Diagnostics.Stopwatch]::StartNew();$failure='';$records=@()
            @{state='running';job=$job.index;episode=$job.episode;mode=$job.mode;port=$using:Port+$slot*4}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
            try{
                & "$using:repo/scripts/run_registered_combat_episodes.ps1" -Plan $job.plan -Port ($using:Port+$slot*4) *>&1|Set-Content "$using:root/jobs/job-$($job.index).log"
                $report=Get-Content (Join-Path $job.root 'report.json') -Raw|ConvertFrom-Json
                if($report.state -ne 'complete' -or $report.records.Count -ne 1){throw 'Incomplete pool cohort'}
                $records=@($report.records)
            }catch{$failure=$_.Exception.Message}
            [pscustomobject]@{job=$job.index;plan_index=$job.plan_index;slot=$slot;port=$using:Port+$slot*4;wall_seconds=$timer.Elapsed.TotalSeconds;error=$failure;records=$records}
            $job=$null
        }
        @{state='idle'}|ConvertTo-Json|Set-Content "$using:root/slot-$slot.json"
    } -ThrottleLimit $slots)
    $unchanged=$sourceFingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $plansUnchanged=$true
    for($p=0;$p -lt $schedules.Count;$p++){
        $jobs=@($results|Where-Object plan_index -eq $p)
        $records=@($jobs|Sort-Object job|ForEach-Object {$_.records})
        $valid=$unchanged -and @($jobs|Where-Object error).Count -eq 0 -and $schedules[$p].sha256 -eq (Get-FileHash $schedules[$p].path).Hash.ToLowerInvariant()
        $plansUnchanged=$plansUnchanged -and $valid
        @{state=$(if($valid){'complete'}else{'failed'});records=$records;pool_root=$root;source_unchanged=$unchanged}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $schedules[$p].schedule.output_root 'report.json') -Encoding utf8NoBOM
    }
    $valid=$unchanged -and $plansUnchanged -and $results.Count -eq $jobIndex -and @($results|Where-Object error).Count -eq 0
    $frames=0;$episodes=0
    foreach($record in @($results|ForEach-Object {$_.records})){
        $batch=Get-Content (Join-Path $record.artifacts 'report.json') -Raw|ConvertFrom-Json
        $episodes+=$batch.usable_captures
        $frames+=($batch.results|Measure-Object actual_game_frames -Sum).Sum
    }
    @{state=$(if($valid){'complete'}else{'failed'});max_instances=$MaxInstances;cohort_workers=4;slots=$slots;wall_seconds=$clock.Elapsed.TotalSeconds;usable_captures=$episodes;actual_game_frames=$frames;aggregate_game_frames_per_second=$frames/$clock.Elapsed.TotalSeconds;source_unchanged=$unchanged;plans=@($schedules|ForEach-Object {@{path=$_.path;sha256=$_.sha256;model_sha256=$_.schedule.model_sha256}});jobs=$results}|ConvertTo-Json -Depth 14|Set-Content "$root/report.json" -Encoding utf8NoBOM
    if(!$valid){throw 'Pool incomplete; inspect cohort errors'}
    Write-Output $root
}finally{
    Set-Content "$root/monitor.stop" -Value 'complete' -Encoding ascii
    if($monitor -and !$monitor.WaitForExit(10000)){throw 'Resource monitor did not finish'}
}
