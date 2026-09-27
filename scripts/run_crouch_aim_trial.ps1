[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29340,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
if (!$OutputRoot) {$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/crouch-aim-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -FriendlyFireTrial -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$reports=@(foreach ($scale in $Timescales) {
    $files=@(Get-ChildItem $OutputRoot -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')
    if ($files.Count -ne 1) {throw "Expected exactly one bot trace for timescale $scale"}
    & "$PSScriptRoot/check_crouch_aim.ps1" -Trace $files[0].FullName
})
$reports | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputRoot 'aim-report.json')
$reports
