[CmdletBinding()]
param([int]$Port=30930)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_grenade_auto.ps1"
$out=Join-Path $repo ('workspace/artifacts/grenade-auto-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$runtime=& "$PSScriptRoot/prepare_grenade_contact_runtime.ps1"
$config=Join-Path $out 'suite.json'
@{version=1;scenarios=@('../../../../scripts/scenarios/base1-grenade-auto-throw.json','../../../../scripts/scenarios/base1-grenade-auto-friend-reject.json');timescales=@(2);repetitions=2;parallelism=2;base_port=$Port;runtime_root=[IO.Path]::GetRelativePath($out,[string]$runtime);tail_frames=5}|ConvertTo-Json -Depth 8|Set-Content $config -Encoding utf8
# Use actual relative paths even if artifact nesting changes.
$cfg=Get-Content $config -Raw|ConvertFrom-Json
$cfg.scenarios=@('base1-grenade-auto-throw','base1-grenade-auto-friend-reject')|ForEach-Object {[IO.Path]::GetRelativePath($out,(Join-Path $repo "scripts/scenarios/$_.json"))}
$cfg|ConvertTo-Json -Depth 8|Set-Content $config -Encoding utf8
$lines=[collections.generic.list[string]]::new()
& "$PSScriptRoot/run_scenario_suite.ps1" -Config $config|ForEach-Object {$lines.Add([string]$_);Write-Host $_}
$line=@($lines|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'}|Select-Object -Last 1)
if(!$line.Count){throw 'Native suite missing'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$'
$suite=Get-Content (Join-Path $Matches[1] 'suite-summary.json') -Raw|ConvertFrom-Json
$results=@()
foreach($run in $suite.results){
    $item=[ordered]@{accepted=$false;artifacts=$run.directory;repeat=$run.repeat;reason='not_checked'}
    try {
        if(!$suite.provenance_valid -or !$run.accepted){throw 'Native scenario/provenance rejected'}
        $definition=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
        $bot=Get-ChildItem $run.directory -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
        $actor=Get-ChildItem $run.directory -Filter '*human-trace.jsonl'|Select-Object -First 1
        $item.scenario=$definition.name
        $item.metrics=Assert-GrenadeAuto @(Get-Content $bot.FullName|ConvertFrom-Json) @(Get-Content $actor.FullName|ConvertFrom-Json) -Reject:($definition.name -match 'friend-reject')
        $item.accepted=$true;$item.reason='accepted'
    }catch{$item.reason=$_.Exception.Message}
    $results+=[pscustomobject]$item
}
$results|ConvertTo-Json -Depth 12|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Host "Automatic grenade: $out/report.json"
if($results.Count -ne 4 -or @($results|Where-Object {!$_.accepted}).Count){throw 'Automatic grenade suite rejected'}
