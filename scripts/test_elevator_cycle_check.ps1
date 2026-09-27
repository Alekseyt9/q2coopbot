[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_cycle.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results[0]
$s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $s.trace_jsonl
$baseline=Assert-ElevatorCycle @($source|ConvertFrom-Json) $c
foreach($case in @('missing_second','no_return','no_lower_wait','lost_upper_support','combat_fixture','frame_gap','fall_velocity','fall_teleport','landing_damage','fall_jump')){
    $rows=@($source|ConvertFrom-Json)
    $done=@($rows|Where-Object elevator -eq completed)
    $secondBoard=($rows|Where-Object {$_.frame -gt $done[0].frame+3 -and $_.elevator -eq 'board'}|Select-Object -First 1).frame
    $fall=$rows|Where-Object frame -eq ($baseline.return_descent.departure_frame+2)|Select-Object -First 1
    switch($case){
        fall_velocity {$fall.self_velocity[2]=0}
        fall_teleport {$fall.self[2]-=20}
        landing_damage {($rows|Where-Object frame -eq $baseline.return_descent.landing_frame).health=99}
        fall_jump {$fall.sent_command.Up=200}
        missing_second {$done[1].elevator='exit'}
        no_return {foreach($r in $rows|Where-Object {$_.frame -gt $done[0].frame+3 -and $_.frame -lt $secondBoard}){foreach($m in $r.movers|Where-Object model -eq 37){$m.origin[2]=0}}}
        no_lower_wait {foreach($r in $rows|Where-Object {$_.frame -gt $done[0].frame+3 -and $_.frame -lt $secondBoard-10}){$r.on_ground=$false}}
        lost_upper_support {$rows[-1].on_ground=$false}
        combat_fixture {$rows[-1]|Add-Member -Force enemies @(@{class='monster_soldier'})}
        frame_gap {$rows=@($rows|Where-Object frame -ne ($done[0].frame+20))}
    }
    $rejected=$false
    try{$null=Assert-ElevatorCycle $rows $c}catch{$rejected=$true}
    if(!$rejected){throw "Accepted invalid cycle: $case"}
    "Rejected: $case"
}

