[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29940,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_closing_door_approach.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/closing-door-approach-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -Map base2 -GameFrames 200 -ActorScenario "$PSScriptRoot/scenarios/base2-closing-door-approach.json" -SynchronizedStart -UnlimitedLoopbackRate -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    $detail=Assert-ClosingDoorApproach @(Get-Content $run.trace_jsonl|ConvertFrom-Json)
    [pscustomobject]@{timescale=$run.timescale;accepted=$detail.accepted;detail=$detail;trace=$run.trace_jsonl}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
