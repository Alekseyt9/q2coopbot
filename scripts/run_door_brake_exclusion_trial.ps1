[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29920,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_door_brake_exclusion.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/door-brake-exclusion-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -DoorPassTrial -TransitionMap base2 -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    $detail=Assert-DoorBrakeExclusion @(Get-Content $run.trace_jsonl|ConvertFrom-Json)
    [pscustomobject]@{timescale=$run.timescale;accepted=$detail.accepted;detail=$detail;trace=$run.trace_jsonl}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
