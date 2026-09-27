[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29670,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_combat.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/light-detection-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$null=New-Item -ItemType Directory -Path $OutputRoot -Force
$report=@()
foreach($mode in @('bsp','zero')){
    $fixture=Get-Content "$PSScriptRoot/scenarios/base1-monster-light-detection.json" -Raw|ConvertFrom-Json
    if($mode -eq 'zero'){$fixture.light=0}
    $fixturePath=Join-Path $OutputRoot "$mode-fixture.json"
    $fixture|ConvertTo-Json -Depth 4|Set-Content $fixturePath
    $trial=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage') -CombatMoveTrial -WeaponSwitchTrial projectile_blaster -ProjectileComparison -ProjectileRange far -ProjectileFixture $fixturePath -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 40 -Timescales $Timescales -Port $Port -OutputRoot $trial
    foreach($run in @(Get-Content (Join-Path $trial 'summary.json') -Raw|ConvertFrom-Json)){
        $rows=@(Get-Content $run.trace_jsonl|ConvertFrom-Json)
        $start=Read-CombatStart $run.combat_log
        $window=@($rows|Where-Object {$_.frame -ge $start -and $_.frame -lt $start+10})
        if($window.Count -ne 10 -or @($window|Where-Object {$_.sent_command.Buttons -band 1}).Count){throw 'Missing quiet observation window'}
        $damage=@(Read-DamageEvents $run.combat_log)
        if(@($damage|Where-Object frame -lt ($start+10)).Count){throw 'Damage before observation window ended'}
        $source=if($mode -eq 'bsp'){'bsp_static'}else{'test_override'}
        $expected=if($mode -eq 'bsp'){34}else{0}
        if(@($window|Where-Object {$_.light_source -ne $source -or $_.sent_command.Light -ne $expected}).Count){throw 'Wrong light source or sample'}
        $applied=@(Get-Content $run.applied_jsonl|ConvertFrom-Json)
        foreach($r in $window){
            $a=@($applied|Where-Object {$_.client_sequence -eq $r.client_sequence -and $_.kind -eq 'new'})
            if($a.Count -ne 1 -or $a[0].command.Light -ne $expected){throw 'Server did not apply client light'}
        }
        $moving=0
        foreach($r in $window){
            $e=@($r.enemies|Where-Object class -eq monster_flyer)
            if($e.Count -ne 1 -or @($r.enemies).Count -ne 1){throw 'Isolated Flyer missing'}
            if([math]::Abs($e[0].origin[0]-1088)+[math]::Abs($e[0].origin[1]-384) -gt 1){$moving++}
        }
        $humanConfig=Get-Content $run.human_config_json -Raw|ConvertFrom-Json
        $human=@(Get-Content $humanConfig.output.trace_jsonl|ConvertFrom-Json|Where-Object {$_.frame -ge $start -and $_.frame -lt $start+10})
        $humanLight=if($mode -eq 'bsp'){58}else{0}
        if($human.Count -ne 10 -or @($human|Where-Object {$_.light_source -ne $source -or $_.sent_command.Light -ne $humanLight -or ($_.sent_command.Buttons -band 1)}).Count){throw 'Actor lighting/control mismatch'}
        $accepted=if($mode -eq 'bsp'){$moving -ge 2}else{$moving -eq 0}
        $report+=[pscustomobject]@{mode=$mode;timescale=$run.timescale;accepted=$accepted;moving_before_fire=$moving;bot_light=$expected;actor_light=$humanLight;trace=$run.trace_jsonl;combat_log=$run.combat_log;scope='native_detection_of_lit_clients_not_dynamic_lights'}
        $report|ConvertTo-Json -Depth 6|Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Native light detection comparison rejected'}
