[CmdletBinding()]
param([int]$Port=30600,[ValidateRange(1,8)][int]$Parallelism=2,[ValidateRange(1,100)][int]$Repetitions=2)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_armed_grenade.ps1"
$out=Join-Path $repo ('workspace/artifacts/armed-grenade-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$runtime=& "$PSScriptRoot/prepare_drop_risk_runtime.ps1"
$scenes=@([IO.Path]::GetRelativePath($out,(Join-Path $repo 'scripts/scenarios/base1-armed-grenade-release.json')))
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
        $item.metrics=Assert-ArmedGrenade $rows $actorRows
        $item.accepted=$true;$item.reason='accepted'
    }catch{$item.reason=$_.Exception.Message}
    $results+=[pscustomobject]$item
}
$results|ConvertTo-Json -Depth 12|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Host "Armed grenade: $out/report.json"
if($results.Count -ne $Repetitions -or @($results|Where-Object {!$_.accepted}).Count){throw 'Armed grenade suite rejected'}


