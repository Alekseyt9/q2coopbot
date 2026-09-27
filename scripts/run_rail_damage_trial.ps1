[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29520,[string]$RuntimeRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage'),[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_rail_damage.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/rail-damage-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$report=@()
foreach($mode in @('rail_precision','rail_friend_behind')) {
    $trial=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot $RuntimeRoot -CombatMoveTrial -WeaponSwitchTrial $mode -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 120 -Timescales $Timescales -Port $Port -OutputRoot $trial
    foreach($run in @(Get-Content (Join-Path $trial 'summary.json') -Raw | ConvertFrom-Json)) {
        $accepted=$false;$detail=$null;$reason=''
        try {
            $events=@(Read-DamageEvents $run.server_log)
            $detail=Assert-RailDamage @(Get-Content $run.trace_jsonl | ConvertFrom-Json) $events $mode
            $accepted=$true
        }catch{$reason=$_.Exception.Message}
        $report += [pscustomobject]@{mode=$mode;timescale=$run.timescale;accepted=$accepted;detail=$detail;reason=$reason;trace=$run.trace_jsonl;server_log=$run.server_log}
        $report | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Rail damage trial rejected'}
