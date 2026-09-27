[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29720,[string]$OutputRoot='',[switch]$FiringStall)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_parasite_spacing.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/parasite-spacing-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$fixture=if($FiringStall){'base1-firing-lane-stall.json'}else{'base1-parasite-spacing.json'}
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -CombatSpacingTrial -CombatSpacingFixture "$PSScriptRoot/scenarios/$fixture" -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    foreach($config in @($run.bot_config_json,$run.human_config_json)){if((Get-Content $config -Raw|ConvertFrom-Json).test.invulnerable){throw 'Damage comparison requires vulnerable clients'}}
    $detail=Measure-ParasiteSpacing @(Get-Content $run.trace_jsonl|ConvertFrom-Json) @(Read-DamageEvents $run.combat_log)
    $accepted=$detail.escape_accepted -and $detail.combat_completed -and $detail.reposition_steps -ge 2
    if($FiringStall){
        # Already outside tongue reach: require recovery, not ten retreat commands.
        $accepted=$detail.combat_completed -and $detail.bot_damage -eq 0 -and $detail.reposition_steps -ge 1
        $rows=@(Get-Content $run.trace_jsonl|ConvertFrom-Json)
        # The initial observe window masks the shot-limit reason, but does not
        # suppress movement. Check the first actually sent reposition command.
        $first=@($rows|Where-Object {$_.arbitration.move_source -eq 'combat_firing_position' -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0)})[0]
        if(!$first -or [math]::Abs($first.self[0]+44.625) -gt 1 -or [math]::Abs($first.self[1]+424.625) -gt 1){throw 'Recorded stall position not exercised'}
        $detail.scope='recorded_wall_friendly_line_stall_native_parasite'
    }
    [pscustomobject]@{timescale=$run.timescale;accepted=$accepted;detail=$detail;trace=$run.trace_jsonl;combat_log=$run.combat_log}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Parasite scene remains incomplete: require escape and completion of combat'}

