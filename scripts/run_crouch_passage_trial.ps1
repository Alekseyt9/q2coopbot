[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29360,[string]$OutputRoot='', [ValidateSet('natural','too-tight')][string[]]$Cases=@('natural','too-tight'))
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_crouch_passage.ps1"
$repo=Split-Path $PSScriptRoot -Parent
if (!$OutputRoot) {$OutputRoot=Join-Path $repo ('workspace/artifacts/crouch-passage-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$report=@()
foreach ($case in $Cases) {
    $scenario=if ($case -eq 'natural') {'base1-low-ceiling-crouch.json'} else {'base1-low-ceiling-too-tight.json'}
    $trial=Join-Path $OutputRoot $case
    & "$PSScriptRoot/run_speed_trial.ps1" -ActorScenario (Join-Path $PSScriptRoot "scenarios/$scenario") -Map base1 -GameFrames 180 -SynchronizedStart -UnlimitedLoopbackRate -Timescales $Timescales -Port $Port -OutputRoot $trial
    foreach ($scale in $Timescales) {
        $file=@(Get-ChildItem $trial -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')
        if ($file.Count -ne 1) {throw 'Expected one bot trace per scale'}
        $rows=@(Get-Content $file[0].FullName | ConvertFrom-Json)
        $accepted=$false; $detail=$null; $reason=''
        try {
            $detail=if ($case -eq 'natural') {Assert-NaturalCrouchPassage $rows} else {Assert-TooTightCrouchRefusal $rows}
            $accepted=$true
        } catch {$reason=$_.Exception.Message}
        $report += [pscustomobject]@{case=$case;timescale=$scale;accepted=$accepted;detail=$detail;reason=$reason;trace=$file[0].FullName}
        $report | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$report
if (@($report | Where-Object {!$_.accepted}).Count) {throw 'Crouch passage not confirmed'}
