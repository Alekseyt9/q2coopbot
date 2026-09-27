[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29480,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_rail_aim.ps1"
if (!$OutputRoot) {$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/rail-aim-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$report=@()
foreach ($mode in @('rail_precision','rail_friend_behind')) {
    $trial=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -CombatMoveTrial -WeaponSwitchTrial $mode -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 120 -Timescales $Timescales -Port $Port -OutputRoot $trial
    foreach ($scale in $Timescales) {
        $files=@(Get-ChildItem $trial -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')
        if ($files.Count -ne 1) {throw 'Expected one rail trace'}
        $accepted=$false;$detail=$null;$reason=''
        try {$detail=Assert-RailAim @(Get-Content $files[0].FullName | ConvertFrom-Json) $mode;$accepted=$true} catch {$reason=$_.Exception.Message}
        $report += [pscustomobject]@{mode=$mode;timescale=$scale;accepted=$accepted;detail=$detail;reason=$reason;trace=$files[0].FullName}
        $report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$report
if (@($report|Where-Object {!$_.accepted}).Count) {throw 'Rail aim trial rejected'}
