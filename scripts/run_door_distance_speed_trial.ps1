[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29950,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_closing_door_phases.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/door-distance-speed-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
New-Item -ItemType Directory -Force $OutputRoot|Out-Null
$OutputRoot=(Resolve-Path $OutputRoot).Path
$runtime=& "$PSScriptRoot/prepare_closing_door_runtime.ps1"
Copy-Item (Join-Path $runtime 'door-fixture.json') (Join-Path $OutputRoot 'door-fixture.json')
$fixture=Get-Content (Join-Path $OutputRoot 'door-fixture.json') -Raw|ConvertFrom-Json
$matrix=Get-Content "$PSScriptRoot/scenarios/base2-door-distance-speed.json" -Raw|ConvertFrom-Json
$results=@()
foreach($case in $matrix.cases){
    $scenario=Get-Content "$PSScriptRoot/scenarios/base2-closing-door-stop.json" -Raw|ConvertFrom-Json
    $scenario.name=$case.name;$scenario.bot_release_frame=$case.release;$scenario.bot_origin[0]=$case.x
    $scenario|Add-Member -NotePropertyName bot_door_pass_speed -NotePropertyValue $case.speed
    $path=Join-Path $OutputRoot ($case.name+'.json')
    $scenario|ConvertTo-Json -Depth 8|Set-Content $path
    $output=Join-Path $OutputRoot $case.name
    & "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot $runtime -Map base2 -GameFrames 200 -ActorScenario $path -SynchronizedStart -UnlimitedLoopbackRate -Timescales $Timescales -Port $Port -OutputRoot $output
    foreach($run in @(Get-Content (Join-Path $output 'summary.json') -Raw|ConvertFrom-Json)){
        if(!(Select-String -LiteralPath $run.server_log -SimpleMatch '.ent file maps/base2.ent loaded.')){throw 'Door fixture not loaded'}
        $detail=Assert-ClosingDoorPhase @(Get-Content $run.trace_jsonl|ConvertFrom-Json) $fixture $case.release $case.expected $case.speed $case.coast_margin $case.x
        $results += [pscustomobject]@{timescale=$run.timescale;name=$case.name;detail=$detail;trace=$run.trace_jsonl}
    }
}
$report=@(foreach($scale in $Timescales){[pscustomobject]@{timescale=$scale;accepted=$true;cases=@($results|Where-Object timescale -eq $scale)}})
$report|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')
$report
