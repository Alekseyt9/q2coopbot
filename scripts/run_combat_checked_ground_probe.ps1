[CmdletBinding()]
param([Parameter(Mandatory)][string]$Protocol,[Parameter(Mandatory)][string]$OutputRoot,
      [Parameter(Mandatory)][string]$Python)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$protocolPath=(Resolve-Path -LiteralPath $Protocol).Path
$spec=Get-Content -LiteralPath $protocolPath -Raw|ConvertFrom-Json
$underfire=$spec.version -eq 'combat_checked_ground_underfire_probe_v1'
$caseCount=$(if($underfire){1}else{3})
$budget=4*$caseCount
if(($spec.version -ne 'combat_checked_ground_native_probe_v1' -and !$underfire) -or $spec.max_instances -ne 16 -or $spec.repeats -ne 2 -or $spec.cases.Count -ne $caseCount -or $spec.new_native_games -ne $budget -or $spec.new_cuda_checks -ne $budget){throw 'Unexpected fixed probe budget'}
if($underfire){foreach($item in $spec.source_files.PSObject.Properties){if((Get-FileHash -LiteralPath $item.Name).Hash.ToLowerInvariant() -ne $item.Value){throw 'Probe source changed'}}}
if((Get-FileHash -LiteralPath $spec.model).Hash.ToLowerInvariant() -ne $spec.model_sha256){throw 'Frozen weights changed'}
if((Get-FileHash -LiteralPath $spec.prerequisite_report).Hash.ToLowerInvariant() -ne $spec.prerequisite_report_sha256){throw 'Prerequisite changed'}
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh probe output required'}
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $spec.port -and $_.LocalPort -lt $spec.port+16}).Count){throw 'Probe ports occupied'}
$null=New-Item -ItemType Directory -Path $root
. "$PSScriptRoot/harness_manifest.ps1"
. "$PSScriptRoot/combat_harness_bundle.ps1"
$sourceFingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$protocolHash=(Get-FileHash -LiteralPath $protocolPath).Hash.ToLowerInvariant()
@{stage='building';runner_pid=$PID;new_native_games=$budget;training_performed=$false}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
$bundle=New-HarnessBinaryBundle $repo
$jobs=@();$index=0
foreach($case in $spec.cases){
    if((Get-FileHash -LiteralPath $case.original_trace).Hash.ToLowerInvariant() -ne $case.original_trace_sha256){throw 'Original trace changed'}
    if((Get-FileHash -LiteralPath $case.reward_config).Hash.ToLowerInvariant() -ne $case.reward_sha256){throw 'Reward changed'}
    $fixture=Join-Path $root ($case.name+'-fixture.json')
    @{instances=@($case.fixture)}|ConvertTo-Json -Depth 25|Set-Content -LiteralPath $fixture -Encoding utf8NoBOM
    foreach($label in @('control','request')){foreach($repeat in 0..1){
        $jobs+=@{job=$index;case_name=$case.name;case=$case;label=$label;repeat=$repeat;fixture=$fixture;
            seed=$case.seed;port=$spec.port+$index;root=(Join-Path $root "$($case.name)-$label-$repeat");frame=$(if($label -eq 'request'){$case.frame}else{0})}
        $index++
    }}
}
try{
    @{stage='native-probe';runner_pid=$PID;cases=$budget}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
    $results=@($jobs|ForEach-Object -Parallel {
        $ErrorActionPreference='Stop';$job=$_;$errorText=''
        try{
            $arguments=@{OutputRoot=$job.root;Workers=1;EpisodesPerWorker=1;Timescale=2;GameFrames=$job.case.game_frames;
                Seed=$job.seed;Port=$job.port;Skill=$job.case.skill;CombatMode='learned';ProviderFile=$using:spec.model;
                Loadout=$job.case.loadout;Synchronous=$true;StopOnGoal=$true;StopOnFirstDeath=$true;ReleaseGameFrame=100;
                Mixed=($job.case.fixture.monsters.Count -gt 1);
                RewardConfig=$job.case.reward_config;GeneratedFixtures=$job.fixture;CheckedWallSlide=$true;BinaryBundle=$using:bundle;
                CheckedGroundRequestProbeFrame=$job.frame}
            & "$using:repo/scripts/run_learned_combat_baseline.ps1" @arguments *>&1|Set-Content "$using:root/job-$($job.job).log"
            $capture=Get-Content -LiteralPath (Join-Path $job.root 'report.json') -Raw|ConvertFrom-Json
            if(!$capture.capture_complete -or !$capture.provenance_valid){throw 'Probe capture incomplete'}
        }catch{$errorText=$_.Exception.Message}
        [pscustomobject]@{job=$job.job;case_name=$job.case_name;label=$job.label;repeat=$job.repeat;seed=$job.seed;
            root=$job.root;frame=$job.frame;port=$job.port;error=$errorText}
    } -ThrottleLimit 16)
    $unchanged=$sourceFingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $valid=$unchanged -and $results.Count -eq $budget -and !@($results|Where-Object error).Count
    @{state=$(if($valid){'complete'}else{'failed'});version=$spec.version;source_unchanged=$unchanged;jobs=$results;
        protocol_sha256=$protocolHash;source_fingerprint=$sourceFingerprint;max_instances=16;timescale=2;
        training_performed=$false;promotion=$false;scope=$spec.scope}|ConvertTo-Json -Depth 12|Set-Content "$root/report.json" -Encoding utf8NoBOM
    if(!$valid){throw 'Native probe failed; preserved attempts require inspection'}
    if((Get-FileHash -LiteralPath $protocolPath).Hash.ToLowerInvariant() -ne $protocolHash){throw 'Protocol changed'}
    @{stage='auditing';runner_pid=$PID}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
    & $Python "$PSScriptRoot/audit_combat_checked_ground_native_probe.py" --protocol $protocolPath --root $root --out "$root/analysis" *> "$root/analysis.log"
    if($LASTEXITCODE){throw 'Native/GPU probe audit failed; inspect analysis log'}
    if($underfire){foreach($item in $spec.source_files.PSObject.Properties){if((Get-FileHash -LiteralPath $item.Name).Hash.ToLowerInvariant() -ne $item.Value){throw 'Probe source changed'}}}
    @{stage='complete';runner_pid=$PID;analysis_complete_sha256=(Get-FileHash "$root/analysis/complete.json").Hash.ToLowerInvariant()}|
        ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
}catch{
    @{stage='failed';runner_pid=$PID;error=$_.Exception.Message}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
    throw
}
