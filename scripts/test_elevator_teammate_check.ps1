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
foreach($case in @('pushing','damage','lost_support','moving_platform','no_completion','actor_teleport')){
 $rows=@($source|ConvertFrom-Json);$human=@($actors|ConvertFrom-Json)
 $r=$rows|Where-Object elevator -eq exit_teammate_wait|Select-Object -First 1
 switch($case){
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
