[CmdletBinding()]
param([int]$Port=30500,[ValidateRange(1,8)][int]$Parallelism=2,[ValidateRange(1,100)][int]$Repetitions=2)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_drop_risk.ps1"
$out=Join-Path $repo ('workspace/artifacts/drop-risk-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$runtime=& "$PSScriptRoot/prepare_drop_risk_runtime.ps1"
$scenes=@('base1-drop-hp100','base1-drop-hp32','base1-drop-hp7','base3-drop-hp100','base3-drop-hp44','base3-drop-hp7'|ForEach-Object {[IO.Path]::GetRelativePath($out,(Join-Path $repo "scripts/scenarios/$_.json"))})
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
        $item.metrics=Assert-DropRisk $rows $definition.bot_health $definition.map
        $item.accepted=$true;$item.reason='accepted'
    }catch{$item.reason=$_.Exception.Message}
    $results+=[pscustomobject]$item
}
$results|ConvertTo-Json -Depth 12|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Host "Drop risk: $out/report.json"
if($results.Count -ne 6*$Repetitions -or @($results|Where-Object {!$_.accepted}).Count){throw 'Drop risk suite rejected'}

