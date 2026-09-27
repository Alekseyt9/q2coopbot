[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_teammate_wait.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$tested=0
foreach($run in $suite.results){
 $c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
 if($c.name -notlike 'base2-elevator-exit-teammate-cross*'){continue}
 $s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
 $source=Get-Content $s.trace_jsonl;$humanSource=Get-Content $s.human_trace_jsonl
 $null=Assert-ElevatorTeammateWait @($source|ConvertFrom-Json) $c @($humanSource|ConvertFrom-Json)
 foreach($case in @('damage','lost_support','overlap','pushing','drift','actor_teleport','no_crossing','bad_completion')){
  $rows=@($source|ConvertFrom-Json);$human=@($humanSource|ConvertFrom-Json)
  $r=$rows|Where-Object elevator -eq exit_teammate_wait|Select-Object -Skip 2 -First 1
  switch($case){
   damage {$r.health=90}
   lost_support {$r.on_ground=$false}
   overlap {$r.self[0]=$r.teammate[0];$r.self[1]=$r.teammate[1]}
   pushing {$r.sent_command.Forward=300}
   drift {$r.self[0]-=2}
   actor_teleport {($human|Where-Object {$_.scenario.step_id -eq 'hold-side'}|Select-Object -First 1).self[0]=100}
   no_crossing {foreach($v in $human){$v.self[1]=1408}}
   bad_completion {if($c.name -like '*hold'){$r.elevator='completed'}else{foreach($v in $rows|Where-Object elevator -eq completed){$v.elevator='exit'}}}
  }
  $rejected=$false;try{$null=Assert-ElevatorTeammateWait $rows $c $human}catch{$rejected=$true}
  if(!$rejected){throw "Invalid crossing accepted: $case"};$tested++
 }
}
if($tested -ne 16){throw "Expected two crossing cases, checked $tested mutations"}
"Rejected $tested invalid crossing traces"
