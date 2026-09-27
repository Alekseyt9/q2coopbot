[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29840,[string]$OutputRoot='',[switch]$Edge,[switch]$East)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_wall_coast_brake.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/wall-coast-brake-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$fixture=if($Edge){'base1-edge-coast-brake.json'}else{'base1-wall-coast-brake.json'}
if($East){if(!$Edge){throw 'East requires Edge'};$fixture='base1-edge-east-coast-brake.json'}
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -CombatSpacingTrial -CombatSpacingFixture "$PSScriptRoot/scenarios/$fixture" -WallBrakingTrial -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 100 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    $detail=Assert-WallCoastBrake @(Get-Content $run.trace_jsonl|ConvertFrom-Json) -Edge:$Edge -East:$East
    [pscustomobject]@{timescale=$run.timescale;accepted=$detail.accepted;detail=$detail;trace=$run.trace_jsonl}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
