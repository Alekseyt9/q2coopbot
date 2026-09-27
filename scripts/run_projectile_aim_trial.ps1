[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29460,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_aim.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/projectile-aim-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$report=@()
foreach($mode in @('projectile_blaster','projectile_hyper')) {
    $trial=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -CombatMoveTrial -WeaponSwitchTrial $mode -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $trial
    foreach($scale in $Timescales) {
        $paths=@(Get-ChildItem $trial -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')
        if($paths.Count -ne 1){throw 'Expected one trace per scale'}
        $accepted=$false;$detail=$null;$reason=''
        try{$detail=Assert-ProjectileAim @(Get-Content $paths[0].FullName | ConvertFrom-Json) $mode;$accepted=$true}catch{$reason=$_.Exception.Message}
        $report += [pscustomobject]@{mode=$mode;timescale=$scale;accepted=$accepted;detail=$detail;reason=$reason;trace=$paths[0].FullName}
        $report | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$report
if(@($report|Where-Object {!$_.accepted}).Count){throw 'Projectile aim trial rejected'}
