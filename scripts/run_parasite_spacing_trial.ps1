[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29720,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_parasite_spacing.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/parasite-spacing-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -CombatSpacingTrial -CombatSpacingFixture "$PSScriptRoot/scenarios/base1-parasite-spacing.json" -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    foreach($config in @($run.bot_config_json,$run.human_config_json)){if((Get-Content $config -Raw|ConvertFrom-Json).test.invulnerable){throw 'Damage comparison requires vulnerable clients'}}
    $detail=Measure-ParasiteSpacing @(Get-Content $run.trace_jsonl|ConvertFrom-Json) @(Read-DamageEvents $run.combat_log)
    [pscustomobject]@{timescale=$run.timescale;accepted=($detail.escape_accepted -and $detail.combat_completed -and $detail.reposition_steps -ge 2);detail=$detail;trace=$run.trace_jsonl;combat_log=$run.combat_log}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Parasite scene remains incomplete: require escape and completion of combat'}

