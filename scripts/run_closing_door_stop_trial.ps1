[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29950,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_closing_door_stop.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/closing-door-stop-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$runtime=& "$PSScriptRoot/prepare_closing_door_runtime.ps1"
& "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot $runtime -Map base2 -GameFrames 200 -ActorScenario "$PSScriptRoot/scenarios/base2-closing-door-stop.json" -SynchronizedStart -UnlimitedLoopbackRate -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
Copy-Item (Join-Path $runtime 'door-fixture.json') (Join-Path $OutputRoot 'door-fixture.json')
$fixture=Get-Content (Join-Path $OutputRoot 'door-fixture.json') -Raw|ConvertFrom-Json
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw|ConvertFrom-Json)){
    if(!(Select-String -LiteralPath $run.server_log -SimpleMatch '.ent file maps/base2.ent loaded.')){throw 'Server did not load door fixture'}
    $detail=Assert-ClosingDoorStop @(Get-Content $run.trace_jsonl|ConvertFrom-Json) $fixture
    [pscustomobject]@{timescale=$run.timescale;accepted=$detail.accepted;detail=$detail;trace=$run.trace_jsonl}
})
$report|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
$report
