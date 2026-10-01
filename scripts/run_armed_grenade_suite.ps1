[CmdletBinding()]
param([int]$Port=30600,[ValidateRange(1,8)][int]$Parallelism=2,[ValidateRange(1,100)][int]$Repetitions=2,[switch]$Contact,[switch]$MovingFriend,[switch]$MovingTarget,[switch]$MovingContact,[switch]$GeometryBlocked)
$ErrorActionPreference='Stop'
if(([int][bool]$Contact+[int][bool]$MovingFriend+[int][bool]$MovingTarget+[int][bool]$MovingContact+[int][bool]$GeometryBlocked) -gt 1){throw 'Select one grenade fixture'}
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_armed_grenade.ps1"
$out=Join-Path $repo ('workspace/artifacts/armed-grenade-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$calibrator=Join-Path $out 'q2grenade-report.exe'
Push-Location $repo
try {
    $env:GOFLAGS='-buildvcs=false'
    $env:GOCACHE=Join-Path $repo 'workspace/build/go-cache'
    go build -o $calibrator ./cmd/q2grenade-report
    if($LASTEXITCODE){throw 'Grenade calibrator build failed'}
} finally {Pop-Location}
$runtime=if($Contact){& "$PSScriptRoot/prepare_grenade_contact_runtime.ps1"}elseif($MovingContact){& "$PSScriptRoot/prepare_grenade_moving_contact_runtime.ps1"}elseif($MovingTarget){& "$PSScriptRoot/prepare_grenade_moving_target_runtime.ps1"}else{& "$PSScriptRoot/prepare_drop_risk_runtime.ps1"}
$scene=if($Contact){'scripts/scenarios/base1-grenade-damageable-contact.json'}elseif($MovingContact){'scripts/scenarios/base1-grenade-moving-contact.json'}elseif($MovingFriend){'scripts/scenarios/base1-grenade-moving-teammate.json'}elseif($MovingTarget){'scripts/scenarios/base1-grenade-moving-target.json'}elseif($GeometryBlocked){'scripts/scenarios/base1-grenade-geometry-blocked.json'}else{'scripts/scenarios/base1-armed-grenade-release.json'}
$scenes=@([IO.Path]::GetRelativePath($out,(Join-Path $repo $scene)))
$config=Join-Path $out 'suite.json'
@{version=1;scenarios=$scenes;timescales=@(2);repetitions=$Repetitions;parallelism=$Parallelism;base_port=$Port;runtime_root=[IO.Path]::GetRelativePath($out,[string]$runtime);tail_frames=5}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
$lines=[collections.generic.list[string]]::new();$failure=''
try{& "$PSScriptRoot/run_scenario_suite.ps1" -Config $config|ForEach-Object {$lines.Add([string]$_);Write-Host $_}}catch{$failure=$_.Exception.Message}
$line=@($lines|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'}|Select-Object -Last 1)
if(!$line.Count){throw $(if($failure){$failure}else{'Suite output missing'})}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$suite=Get-Content (Join-Path $Matches[1] 'suite-summary.json') -Raw|ConvertFrom-Json
$results=@()
foreach($run in $suite.results){
    $item=[ordered]@{accepted=$false;repeat=$run.repeat;port=$run.port;artifacts=$run.directory;reason='not_checked'}
    try{
        if($failure -or !$suite.provenance_valid -or !$run.accepted){throw $(if($failure){$failure}else{'Native scenario or provenance rejected'})}
        $definition=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
        $file=Get-ChildItem $run.directory -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
        if(!$file){throw 'Bot trace missing'}
        $rows=@(Get-Content $file.FullName|ConvertFrom-Json)
        $actor=Get-ChildItem $run.directory -Filter '*human-trace.jsonl'|Select-Object -First 1
        if(!$actor){throw 'Teammate trace missing'}
        $actorRows=@(Get-Content $actor.FullName|ConvertFrom-Json)
        if($GeometryBlocked){$item.geometry=Assert-GrenadeGeometry $rows $actorRows -Blocked}elseif($MovingTarget){$item.metrics=Assert-GrenadeMovingTarget $rows $actorRows;$item.envelope=Assert-GrenadeEarlyEnvelope $rows -Kind target}elseif($MovingFriend){$item.metrics=Assert-GrenadeMovingFriend $rows $actorRows;$item.envelope=Assert-GrenadeEarlyEnvelope $rows;$item.geometry=Assert-GrenadeGeometry $rows $actorRows}else{
        $item.metrics=Assert-ArmedGrenade $rows $actorRows -Contact:($Contact -or $MovingContact)
        $calibration=Join-Path $run.directory 'grenade-calibration.json'
        $calibratorArgs=@('-trace',$file.FullName,'-root',(Join-Path $runtime 'baseq2'),'-map','base1','-out',$calibration)
        if($Contact){$calibratorArgs+='-contact'}
        if($MovingContact){$calibratorArgs+='-moving-contact'}
        & $calibrator @calibratorArgs
        if($LASTEXITCODE){throw 'Observed native grenade path disagrees with prediction'}
        $item.calibration=Get-Content $calibration -Raw|ConvertFrom-Json
        }
        $item.accepted=$true;$item.reason='accepted'
    }catch{$item.reason=$_.Exception.Message}
    $results+=[pscustomobject]$item
}
$results|ConvertTo-Json -Depth 12|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Host "Armed grenade: $out/report.json"
if($results.Count -ne $Repetitions -or @($results|Where-Object {!$_.accepted}).Count){throw 'Armed grenade suite rejected'}


