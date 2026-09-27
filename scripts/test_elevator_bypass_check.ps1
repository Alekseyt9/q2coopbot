[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_teammate_wait.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results[0];$s=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$c=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $s.trace_jsonl;$human=@(Get-Content $s.human_trace_jsonl|ConvertFrom-Json)
$null=Assert-ElevatorTeammateWait @($source|ConvertFrom-Json) $c $human
foreach($case in @('damage','lost_support','overlap','excessive_side','moved_blocker','no_completion')){
 $rows=@($source|ConvertFrom-Json);$r=$rows|Where-Object elevator -eq exit_teammate_bypass|Select-Object -Skip 2 -First 1
 switch($case){
 damage {$r.health=90}
 lost_support {$r.on_ground=$false}
 overlap {$r.self[0]=$r.teammate[0];$r.self[1]=$r.teammate[1]}
 excessive_side {$r.self[1]=1460}
 moved_blocker {$r.teammate[0]=100}
 no_completion {foreach($v in $rows|Where-Object elevator -eq completed){$v.elevator='exit'}}
 }
 $rejected=$false;try{$null=Assert-ElevatorTeammateWait $rows $c $human}catch{$rejected=$true}
 if(!$rejected){throw "Invalid bypass accepted: $case"};"Rejected: $case"
}
