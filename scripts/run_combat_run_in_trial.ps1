[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29800,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_combat_run_in.ps1"
. "$PSScriptRoot/check_ground_prediction.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/combat-run-in-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -CombatSpacingTrial -CombatSpacingFixture "$PSScriptRoot/scenarios/base1-parasite-run-in.json" -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 220 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    foreach($config in @($run.bot_config_json,$run.human_config_json)){if((Get-Content $config -Raw|ConvertFrom-Json).test.invulnerable){throw 'Run-in requires vulnerable clients'}}
    $detail=Measure-CombatRunIn @(Get-Content $run.trace_jsonl|ConvertFrom-Json) @(Read-DamageEvents $run.combat_log)
    $prediction=Assert-GroundPredictionRunIn @(Get-Content $run.trace_jsonl|ConvertFrom-Json)
    $detail|Add-Member -NotePropertyName ground_prediction -NotePropertyValue $prediction
    [pscustomobject]@{timescale=$run.timescale;accepted=$detail.accepted;detail=$detail;trace=$run.trace_jsonl;combat_log=$run.combat_log}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Require measured incoming speed, braking, escape and kill without damage'}
