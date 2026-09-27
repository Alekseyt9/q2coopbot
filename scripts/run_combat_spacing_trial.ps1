[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29700,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_combat_spacing.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/combat-spacing-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -CombatSpacingTrial -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 100 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    $detail=Assert-CombatSpacing @(Get-Content $run.trace_jsonl|ConvertFrom-Json)
    [pscustomobject]@{timescale=$run.timescale;accepted=$true;detail=$detail;trace=$run.trace_jsonl}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
