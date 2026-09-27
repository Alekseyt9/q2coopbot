[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_teammate_wait.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results|Where-Object { (Get-Content (Join-Path $_.directory 'scenario.json') -Raw|ConvertFrom-Json).name -like '*release' }|Select-Object -First 1
$s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $s.trace_jsonl;$actors=Get-Content $s.human_trace_jsonl
$null=Assert-ElevatorTeammateWait @($source|ConvertFrom-Json) $c @($actors|ConvertFrom-Json)
$cases=@('pushing','damage','lost_support','moving_platform','no_completion','actor_teleport')
if($c.name -like '*side-walls*'){$cases+=@('missing_wall','shifted_wall','unsafe_bypass')}
if($c.name -like '*ascent*'){$cases+=@('early_wait','no_ascent_blocker','no_retreat')}
foreach($case in $cases){
 $rows=@($source|ConvertFrom-Json);$human=@($actors|ConvertFrom-Json)
 $r=$rows|Where-Object elevator -eq exit_teammate_wait|Select-Object -First 1
 switch($case){
 early_wait {($rows|Where-Object {$_.map -eq 'base2' -and $_.frame -eq 70}).elevator='exit_teammate_wait'}
 no_ascent_blocker {foreach($v in $rows|Where-Object {$_.map -eq 'base2' -and $_.frame -lt 75}){$v|Add-Member -NotePropertyName teammate -NotePropertyValue @(100,1408,24.125) -Force}}
 no_retreat {foreach($v in $rows|Where-Object elevator -eq exit_teammate_retreat){$v.elevator='landing_probe'}}
 missing_wall {$r.movers=@($r.movers|Where-Object {$_.model -ne 1 -or $_.origin[0] -ne -576})}
 shifted_wall {foreach($wall in $r.movers|Where-Object {$_.model -eq 1 -and $_.origin[0] -eq -576}){$wall.origin[1]+=100}}
 unsafe_bypass {$r.elevator='exit_teammate_bypass'}
 pushing {$r.sent_command.Forward=300}
 damage {$r.health=90}
 lost_support {$r.on_ground=$false}
 moving_platform {($r.movers|Where-Object model -eq 50).origin[2]=-10}
 no_completion {foreach($v in $rows|Where-Object elevator -eq completed){$v.elevator='exit'}}
 actor_teleport {($human|Where-Object {$_.scenario.step_id -eq 'clear-lift-exit'}|Select-Object -Skip 1 -First 1).self[0]+=80}
 }
 $rejected=$false;try{$null=Assert-ElevatorTeammateWait $rows $c $human}catch{$rejected=$true}
 if(!$rejected){throw "Invalid waiting accepted: $case"};"Rejected: $case"
}
