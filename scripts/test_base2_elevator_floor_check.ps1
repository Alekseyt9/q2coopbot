[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_base2_elevator_floor.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results[0];$s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $s.trace_jsonl
$null=Assert-Base2ElevatorFloor @($source|ConvertFrom-Json) $c
foreach($case in @('floor_drift','lost_support','frozen_platform','damage','no_completion')){
 $rows=@($source|ConvertFrom-Json)
 $r=$rows|Where-Object {$_.map -eq 'base2' -and $_.elevator -eq 'ride'}|Select-Object -Last 1
 switch($case){
 floor_drift {($rows|Where-Object {$_.map -eq 'base2' -and $_.frame -eq $c.bot_release_frame}).self[0]+=5}
 lost_support {$r.on_ground=$false}
 frozen_platform {($r.movers|Where-Object model -eq 50).origin[2]=-190}
 damage {$r.health=90}
 no_completion {foreach($row in $rows|Where-Object elevator -eq completed){$row.elevator='exit'}}
 }
 $rejected=$false;try{$null=Assert-Base2ElevatorFloor $rows $c}catch{$rejected=$true}
 if(!$rejected){throw "Invalid floor approach accepted: $case"};"Rejected: $case"
}
