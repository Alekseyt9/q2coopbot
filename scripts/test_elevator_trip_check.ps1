[CmdletBinding()]
param([Parameter(Mandatory)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_elevator_trip.ps1"
$suite=Get-Content (Join-Path $SuiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$run=$suite.results[0]
$summary=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
$scenario=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
$source=Get-Content $summary.trace_jsonl
$rows=@($source|ConvertFrom-Json)
$null=Assert-ElevatorTrip $rows $scenario
$done=($rows|Where-Object elevator -eq completed|Select-Object -First 1).frame
$cases=@('lost_support','damage','missing_frame','frozen_platform','off_edge','wrong_landing','no_clear_exit')
foreach($case in $cases){
    $rows=@($source|ConvertFrom-Json)
    $r=$rows|Where-Object {$_.map -eq 'base3' -and $_.frame -eq $done-3}|Select-Object -First 1
    switch($case){
        lost_support {$r.on_ground=$false}
        damage {$r.health=99}
        missing_frame {$rows=@($rows|Where-Object { !($_.map -eq 'base3' -and $_.frame -eq $done-3) })}
        frozen_platform {($r.movers|Where-Object model -eq 37).origin[2]=-190}
        off_edge {$r.self[0]=900}
        wrong_landing {($rows|Where-Object {$_.map -eq 'base3' -and $_.frame -eq $done+2}).self[2]=-240}
        no_clear_exit {($rows|Where-Object {$_.map -eq 'base3' -and $_.frame -eq $done+3}).self[1]=275}
    }
    $rejected=$false
    try {$null=Assert-ElevatorTrip $rows $scenario} catch {$rejected=$true}
    if(!$rejected){throw "Invalid trace accepted: $case"}
    Write-Output "Rejected: $case"
}
