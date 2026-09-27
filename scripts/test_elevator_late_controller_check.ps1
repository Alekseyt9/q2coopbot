param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_teammate_wait.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results[0];$s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $s.trace_jsonl;$actorSource=Get-Content $s.human_trace_jsonl
$null=Assert-ElevatorLateController @($source|ConvertFrom-Json) $c @($actorSource|ConvertFrom-Json)
foreach($case in @('damage','wait_bottom','no_completion','bad_landing','no_shared_ride','no_actor_walk')){
 $rows=@($source|ConvertFrom-Json);$actor=@($actorSource|ConvertFrom-Json)
 $done=$rows|Where-Object elevator -eq completed|Select-Object -First 1
 switch($case){
 damage {$done.health=90}
 wait_bottom {$done.elevator='wait_bottom'}
 no_completion {$done.elevator='landing_probe'}
 bad_landing {($rows|Where-Object frame -eq ($done.frame+20)).self[0]=-3}
 no_shared_ride {foreach($r in $rows|Where-Object goal -eq cover_teammate){$r.goal='follow_teammate'}}
 no_actor_walk {foreach($r in $actor|Where-Object {$_.scenario.step_id -eq 'leave-platform'}){$r.self[0]=0}}
 }
 $failed=$false;try{$null=Assert-ElevatorLateController $rows $c $actor}catch{$failed=$true}
 if(!$failed){throw "Invalid late-controller trace accepted: $case"};"Rejected: $case"
}
