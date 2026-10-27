[CmdletBinding()]
param([Parameter(Mandatory)][string]$Protocol,[Parameter(Mandatory)][string]$OutputRoot,[switch]$DryRun)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$protocolPath=(Resolve-Path -LiteralPath $Protocol).Path
$spec=Get-Content -LiteralPath $protocolPath -Raw|ConvertFrom-Json
$count=@($spec.cases).Count
if($spec.version -ne 'combat_rocket_surface_native_probe_v1' -or $count -lt 1 -or $count -gt 4 -or $spec.repeats -ne 2 -or $spec.max_instances -ne 16 -or $spec.native_games -ne $count*8){throw 'Invalid rocket probe budget'}
foreach($entry in $spec.inputs.PSObject.Properties){if((Get-FileHash -LiteralPath $entry.Name).Hash.ToLowerInvariant() -ne $entry.Value){throw 'Probe input changed'}}
$names=[Collections.Generic.HashSet[string]]::new()
foreach($case in $spec.cases){
    if($case.name -notmatch '^[a-zA-Z0-9_-]+$' -or !$names.Add([string]$case.name) -or $case.seed -le 0 -or $case.fixture.engine_seed -ne $case.seed -or $case.frame -le 0 -or $case.frame -gt 1000000){throw 'Invalid rocket case'}
    foreach($kind in @('body','floor','wall')){
        $point=$case.points.$kind
        if(@($point).Count -ne 3){throw 'Missing point'}
        foreach($v in $point){if([double]::IsNaN([double]$v) -or [double]::IsInfinity([double]$v) -or [Math]::Abs([double]$v) -gt 8192){throw 'Invalid point coordinate'}}
    }
}
if($spec.port -lt 1024 -or $spec.port+$spec.native_games -gt 65536){throw 'Invalid port range'}
if($DryRun){Write-Output "Verified $count independent worlds, $($spec.native_games) native diagnostic runs";return}
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh output required'}
if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $spec.port -and $_.LocalPort -lt $spec.port+$spec.native_games}).Count){throw 'Probe ports occupied'}
$null=New-Item -ItemType Directory -Path $root
. "$PSScriptRoot/harness_manifest.ps1"
. "$PSScriptRoot/combat_harness_bundle.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$bundle=New-HarnessBinaryBundle $repo
$jobs=@();$index=0
foreach($case in $spec.cases){
    $fixture=Join-Path $root ($case.name+'-fixture.json')
    @{instances=@($case.fixture)}|ConvertTo-Json -Depth 25|Set-Content -LiteralPath $fixture -Encoding utf8NoBOM
    foreach($label in @('source','body','floor','wall')){foreach($repeat in 0..1){
        $point=if($label -eq 'source'){''}else{(@($case.points.$label)|ForEach-Object {([double]$_).ToString('R',[Globalization.CultureInfo]::InvariantCulture)}) -join ','}
        $jobs+=@{job=$index;case_name=$case.name;case=$case;label=$label;repeat=$repeat;fixture=$fixture;point=$point;
            seed=$case.seed;port=$spec.port+$index;root=(Join-Path $root "$($case.name)-$label-$repeat")}
        $index++
    }}
}
$results=@($jobs|ForEach-Object -Parallel {
    $ErrorActionPreference='Stop';$job=$_;$errorText=''
    try {
        $arguments=@{OutputRoot=$job.root;Workers=1;EpisodesPerWorker=1;Timescale=2;GameFrames=$job.case.game_frames;
            Seed=$job.seed;Port=$job.port;Skill=$job.case.fixture.skill;CombatMode='learned';ProviderFile=$job.case.model;
            Loadout=$job.case.fixture.loadout;Synchronous=$true;StopOnGoal=$true;StopOnFirstDeath=$true;ReleaseGameFrame=100;
            Mixed=($job.case.fixture.monsters.Count -gt 1);RewardConfig=$job.case.reward_config;
            GeneratedFixtures=$job.fixture;CheckedWallSlide=$true;BinaryBundle=$using:bundle;
            ExecutionContract='stock_weapons_fire_v1';RequireNativeObjective=$true}
        if($job.label -ne 'source'){$arguments.RocketAimProbeFrame=$job.case.frame;$arguments.RocketAimProbePoint=$job.point}
        & "$using:PSScriptRoot/run_learned_combat_baseline.ps1" @arguments *> (Join-Path $using:root ("job-"+$job.job+'.log'))
    }catch{$errorText=$_.ToString()}
    @{job=$job.job;case_name=$job.case_name;label=$job.label;repeat=$job.repeat;seed=$job.seed;root=$job.root;error=$errorText}
} -ThrottleLimit 16)
$unchanged=(Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -eq $fingerprint
$valid=$unchanged -and $results.Count -eq $jobs.Count -and !@($results|Where-Object {$_.error}).Count
@{state=$(if($valid){'complete'}else{'failed'});source_unchanged=$unchanged;protocol_sha256=(Get-FileHash $protocolPath).Hash.ToLowerInvariant();jobs=$results;training_performed=$false;ordinary_ppo_eligible=$false}|ConvertTo-Json -Depth 20|Set-Content "$root/report.json" -Encoding utf8NoBOM
if(!$valid){throw 'Rocket aim probe incomplete; inspect job errors'}
Write-Output $root
