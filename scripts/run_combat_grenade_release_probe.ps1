[CmdletBinding()]
param([Parameter(Mandatory)][string]$Protocol,[Parameter(Mandatory)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$protocolPath=(Resolve-Path -LiteralPath $Protocol).Path
$spec=Get-Content -LiteralPath $protocolPath -Raw|ConvertFrom-Json
if($spec.version -ne 'combat_grenade_release_native_probe_v1' -or $spec.cases.Count -ne 4 -or $spec.repeats -ne 2 -or $spec.max_instances -ne 16 -or $spec.new_native_games -ne 16){throw 'Unexpected fixed diagnostic budget'}
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh probe output required'}
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $spec.port -and $_.LocalPort -lt $spec.port+16}).Count){throw 'Probe ports occupied'}
foreach($entry in $spec.inputs.PSObject.Properties){if((Get-FileHash -LiteralPath $entry.Name).Hash.ToLowerInvariant() -ne $entry.Value){throw 'Pinned probe input changed'}}
$null=New-Item -ItemType Directory -Path $root
. "$PSScriptRoot/harness_manifest.ps1"
. "$PSScriptRoot/combat_harness_bundle.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$protocolHash=(Get-FileHash -LiteralPath $protocolPath).Hash.ToLowerInvariant()
@{stage='building';runner_pid=$PID}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
$bundle=New-HarnessBinaryBundle $repo
$jobs=@();$index=0
foreach($case in $spec.cases){
    $fixture=Join-Path $root ($case.name+'-fixture.json')
    @{instances=@($case.fixture)}|ConvertTo-Json -Depth 25|Set-Content -LiteralPath $fixture -Encoding utf8NoBOM
    foreach($label in @('source','release')){foreach($repeat in 0..1){
        $jobs+=@{job=$index;case_name=$case.name;case=$case;label=$label;repeat=$repeat;fixture=$fixture;
            seed=$case.seed;port=$spec.port+$index;root=(Join-Path $root "$($case.name)-$label-$repeat");
            frame=$(if($label -eq 'release'){$case.frame}else{0})}
        $index++
    }}
}
try{
    @{stage='native-probe';runner_pid=$PID;cases=16}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
    $results=@($jobs|ForEach-Object -Parallel {
        $ErrorActionPreference='Stop';$job=$_;$errorText=''
        try{
            $arguments=@{OutputRoot=$job.root;Workers=1;EpisodesPerWorker=1;Timescale=2;GameFrames=$job.case.game_frames;
                Seed=$job.seed;Port=$job.port;Skill=$job.case.skill;CombatMode='learned';ProviderFile=$job.case.model;
                Loadout=$job.case.loadout;Synchronous=$true;StopOnGoal=$true;StopOnFirstDeath=$true;ReleaseGameFrame=100;
                Mixed=($job.case.fixture.monsters.Count -gt 1);RewardConfig=$job.case.reward_config;
                GeneratedFixtures=$job.fixture;CheckedWallSlide=$true;BinaryBundle=$using:bundle;
                GrenadeReleaseProbeFrame=$job.frame;ExecutionContract='stock_weapons_fire_v1';RequireNativeObjective=$true}
            & "$using:repo/scripts/run_learned_combat_baseline.ps1" @arguments *>&1|Set-Content "$using:root/job-$($job.job).log"
            $capture=Get-Content -LiteralPath (Join-Path $job.root 'report.json') -Raw|ConvertFrom-Json
            if(!$capture.capture_complete -or !$capture.provenance_valid){throw 'Capture incomplete'}
        }catch{$errorText=$_.Exception.Message}
        [pscustomobject]@{job=$job.job;case_name=$job.case_name;label=$job.label;repeat=$job.repeat;seed=$job.seed;
            root=$job.root;frame=$job.frame;port=$job.port;error=$errorText}
    } -ThrottleLimit 16)
    $unchanged=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $valid=$unchanged -and $results.Count -eq 16 -and !@($results|Where-Object error).Count
    @{state=$(if($valid){'complete'}else{'failed'});source_unchanged=$unchanged;jobs=$results;
        protocol_sha256=$protocolHash;source_fingerprint=$fingerprint;max_instances=16;timescale=2;
        training_performed=$false;promotion=$false;scope=$spec.scope}|ConvertTo-Json -Depth 12|Set-Content "$root/report.json" -Encoding utf8NoBOM
    if(!$valid){throw 'Native probe failed; inspect preserved job receipts'}
    foreach($entry in $spec.inputs.PSObject.Properties){if((Get-FileHash -LiteralPath $entry.Name).Hash.ToLowerInvariant() -ne $entry.Value){throw 'Pinned probe input changed'}}
    @{stage='complete';runner_pid=$PID;native_only=$true}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
}catch{
    @{stage='failed';runner_pid=$PID;error=$_.Exception.Message}|ConvertTo-Json|Set-Content "$root/progress.json" -Encoding utf8NoBOM
    throw
}
