[CmdletBinding()]
param([string]$Config="$PSScriptRoot/scenarios/base2-elevator-teammate-suite.json",[int[]]$Timescales=@(),[int]$Port=0,[string]$OutputRoot='',[switch]$SideWalls)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_teammate_wait.ps1"
if($SideWalls -and !$PSBoundParameters.ContainsKey('Config')){$Config="$PSScriptRoot/scenarios/base2-elevator-side-walls-suite.json"}
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2 -SideWalls:$SideWalls | Out-Null
if($Timescales.Count -or $Port -or $OutputRoot){
    $original=(Resolve-Path $Config).Path;$base=Split-Path $original -Parent
    $cfg=Get-Content $original -Raw|ConvertFrom-Json
    $cfg.scenarios=@($cfg.scenarios|ForEach-Object {(Resolve-Path (Join-Path $base $_)).Path})
    $cfg.runtime_root=(Resolve-Path (Join-Path $base $cfg.runtime_root)).Path
    if($Timescales.Count){$cfg.timescales=$Timescales}
    if($Port){$cfg.base_port=$Port}
    if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/trip-wrapper-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
    New-Item -ItemType Directory -Force $OutputRoot|Out-Null
    $Config=Join-Path (Resolve-Path $OutputRoot).Path 'trip-suite.json'
    $newBase=Split-Path $Config -Parent
    $cfg.scenarios=@($cfg.scenarios|ForEach-Object {[IO.Path]::GetRelativePath($newBase,$_ )})
    $cfg.runtime_root=[IO.Path]::GetRelativePath($newBase,$cfg.runtime_root)
    $cfg|ConvertTo-Json -Depth 6|Set-Content $Config
}
$messages=@(& "$PSScriptRoot/run_scenario_suite.ps1" -Config $Config)
$messages|Write-Output
$line=@($messages|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'})
if($line.Count -ne 1){throw 'Suite output directory unavailable'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$root=$Matches[1]
$suite=Get-Content (Join-Path $root 'suite-summary.json') -Raw|ConvertFrom-Json
if(!$suite.provenance_valid){throw 'Suite inputs changed'}
$details=@(foreach($run in $suite.results){
    $speed=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
    $scenario=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
    $detail=Assert-ElevatorTeammateWait @(Get-Content $speed.trace_jsonl|ConvertFrom-Json) $scenario @(Get-Content $speed.human_trace_jsonl|ConvertFrom-Json)
    [pscustomobject]@{name=$scenario.name;timescale=$run.timescale;port=$run.port;detail=$detail;trace=$speed.trace_jsonl}
})
$details|ConvertTo-Json -Depth 6|Set-Content (Join-Path $root 'trip-report.json')
if($OutputRoot){
    $reports=@(foreach($scale in @($details.timescale|Sort-Object -Unique)){
        [pscustomobject]@{timescale=$scale;accepted=$true;suite=$root;cases=@($details|Where-Object timescale -eq $scale)}
    })
    $reports|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')
}
$details




