[CmdletBinding()]
param([int]$Port=31160)
$ErrorActionPreference='Stop'
if($Port -lt 1024 -or $Port -gt 65532){throw 'Shared checkpoint suite requires three valid ports'}
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
. "$PSScriptRoot/checkpoint_branch_metrics.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$source=& "$PSScriptRoot/prepare_registered_pickup_runtime.ps1" -Profile inventory
$out=Join-Path $repo ('workspace/artifacts/checkpoint-branches-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$client=Join-Path $out 'q2coopbot.exe';$tool=Join-Path $out 'q2checkpoint.exe'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'};go build -o $tool ./cmd/q2checkpoint;if($LASTEXITCODE){throw 'Checkpoint build failed'}}finally{Pop-Location}
$trialScript=Join-Path $PSScriptRoot 'run_checkpoint_trial.ps1';$trialHost=(Get-Process -Id $PID).Path
$seedConfig=Join-Path $out 'seed-config.json'
@{runtime_root=$source;client_exe=$client;checkpoint_exe=$tool;output_root=(Join-Path $out 'seed');port=$Port;timescale=2;barrier=$true}|ConvertTo-Json|Set-Content $seedConfig -Encoding utf8
& $trialHost -NoProfile -File $trialScript -Config $seedConfig|Out-Host
if($LASTEXITCODE){throw 'Common checkpoint capture failed'}
$seed=Get-Content (Join-Path $out 'seed/report.json') -Raw|ConvertFrom-Json
if(!$seed.accepted -or $seed.reason -ne 'accepted' -or $seed.proof -ne 'coordinated_capture_save_and_release'){throw 'Common checkpoint capture unproved'}
$shared=Join-Path $out 'seed/checkpoint'
$records=Get-CheckpointPackageRecords $shared
$packageJSON=$records|ConvertTo-Json -Compress -Depth 4
$packageFingerprint=Get-HarnessFingerprint $records
$configs=@()
foreach($i in 0..1){$path=Join-Path $out "branch-$i.json";@{runtime_root=(Join-Path $out 'seed/runtime');client_exe=$client;checkpoint_exe=$tool;output_root=(Join-Path $out "branch-$i");port=($Port+$i+1);timescale=2;barrier=$true;restore_slots=$true;resume=$true;bot_mode=$(if($i -eq 0){'resume'}else{'fresh'});source_checkpoint=$shared}|ConvertTo-Json|Set-Content $path -Encoding utf8;$configs+=$path}
$results=@($configs|ForEach-Object -Parallel {
    $err='';try{& $using:trialHost -NoProfile -File $using:trialScript -Config $_|Out-Host;if($LASTEXITCODE){$err='Branch process failed'}}catch{$err=$_.Exception.Message}
    $cfg=Get-Content $_ -Raw|ConvertFrom-Json;$path=Join-Path $cfg.output_root 'report.json'
    if(Test-Path $path){$r=Get-Content $path -Raw|ConvertFrom-Json;if($err -or $r.reason -ne 'accepted' -or $r.proof -ne 'new_process_resume_and_fresh'){$r.accepted=$false};$r}else{[pscustomobject]@{accepted=$false;reason=$err;port=$cfg.port}}
} -ThrottleLimit 2)
$accepted=$results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
$copiesMatch=$true;$metrics=@()
if($accepted){
    foreach($i in 0..1){
        $dir=Join-Path $out "branch-$i";$r=Get-Content (Join-Path $dir 'report.json') -Raw|ConvertFrom-Json
        if((Get-CheckpointPackageRecords (Join-Path $dir 'checkpoint')|ConvertTo-Json -Compress -Depth 4) -ne $packageJSON -or $r.barrier.id -ne $seed.barrier.id){$copiesMatch=$false}
        $m=Get-CheckpointBranchMetrics -Trace (Join-Path $dir 'restored-bot.jsonl') -Generation $r.restored_bot.spawncount -FirstFrame $r.bot_receipt.frame -LastFrame $r.completed_actor.scenario.end_frame
        $metrics+=@{branch=$i;mode=$r.bot_mode;package_fingerprint=$packageFingerprint;runner_elapsed=$r.actor_receipt.runner_elapsed;runner_remaining_frames=(500-$r.actor_receipt.runner_elapsed);wall_seconds=$r.branch_wall_seconds;bot=$m}
    }
}
$immutable=(Get-CheckpointPackageRecords $shared|ConvertTo-Json -Compress -Depth 4) -eq $packageJSON
$isolated=$false
if($accepted -and $copiesMatch -and $immutable){
    # Explicit mutation probe after all processes have stopped. Restore the
    # private file afterwards so the retained branch package stays usable.
    $probe=Join-Path $out 'branch-0/checkpoint/native/base2.sav';$original=[IO.File]::ReadAllBytes($probe)
    try{
        $stream=[IO.File]::Open($probe,[IO.FileMode]::Append);try{$stream.WriteByte(0)}finally{$stream.Dispose()}
        $isolated=(Get-CheckpointPackageRecords $shared|ConvertTo-Json -Compress -Depth 4) -eq $packageJSON -and (Get-CheckpointPackageRecords (Join-Path $out 'branch-1/checkpoint')|ConvertTo-Json -Compress -Depth 4) -eq $packageJSON -and (Get-CheckpointPackageRecords (Join-Path $out 'branch-0/checkpoint')|ConvertTo-Json -Compress -Depth 4) -ne $packageJSON
    }finally{[IO.File]::WriteAllBytes($probe,$original)}
}
$provenance=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
$accepted=$accepted -and $copiesMatch -and $immutable -and $isolated -and $provenance
@{accepted=$accepted;proof='shared_checkpoint_independent_native_branches';timescale=2;parallelism=2;source_fingerprint=$fingerprint;provenance_valid=$provenance;package_fingerprint=$packageFingerprint;source_immutable=$immutable;copies_match=$copiesMatch;mutation_isolated=$isolated;source_checkpoint=$shared;results=$results;metrics=$metrics}|ConvertTo-Json -Depth 40|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Output "Checkpoint branches: $out"
if(!$accepted){throw 'Shared checkpoint branches rejected'}
