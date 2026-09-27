[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29410,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
if (!$OutputRoot) {$OutputRoot=Join-Path $repo ('workspace/artifacts/weapon-economy-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
New-Item -ItemType Directory -Path $OutputRoot -Force|Out-Null
$results=@()
foreach ($mode in @('economy_weak','economy_armed')) {
    $out=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -CombatMoveTrial -WeaponSwitchTrial $mode -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 120 -Timescales $Timescales -Port $Port -OutputRoot $out
    foreach ($scale in $Timescales) {
        $path=@(Get-ChildItem $out -Filter "scale-$scale-port-*-trace.jsonl"|Where-Object Name -NotLike '*human*')[0].FullName
        $rows=@(Get-Content $path|ConvertFrom-Json)
        $expected=if ($mode -eq 'economy_weak') {'use Blaster'} else {'use Shotgun'}
        $reason=if ($mode -eq 'economy_weak') {'save_ammo_weak_target'} else {'armed_target'}
        $request=@($rows|Where-Object { $_.weapon_request -eq $expected -and $_.weapon_reason -eq $reason -and $_.inventory_known -and $_.inventory_age_frames -le 20 }|Select-Object -First 1)
        $firing=@($rows|Where-Object {
            $request.Count -and $_.frame -gt $request[0].frame -and ($_.sent_command.Buttons -band 1) -and $_.health -gt 0 -and
            (($mode -eq 'economy_weak' -and $_.weapon -eq 'Blaster') -or ($mode -eq 'economy_armed' -and $_.weapon -like '*/v_shotg/*' -and $_.ammo -gt 0))
        }|Select-Object -First 1)
        $stock=if ($mode -eq 'economy_weak') {'Bullets'} else {'Shells'}
        $samples=@($rows|Where-Object {$_.inventory_known -and $_.inventory_age_frames -le 20 -and $request.Count -and $_.frame -ge $request[0].frame})
        $counts=@($samples|ForEach-Object {$entry=@($_.inventory|Where-Object name -eq $stock);if ($entry.Count) {$entry[0].count} else {0}})
        $stockOK=$counts.Count -gt 0
        if ($stockOK) {
            if ($mode -eq 'economy_weak') {$stockOK=($counts|Measure-Object -Minimum).Minimum -eq 100}
            else {$stockOK=($counts|Measure-Object -Maximum).Maximum -eq 20 -and ($counts|Measure-Object -Minimum).Minimum -lt 20}
        }
        $requests=@($rows|Where-Object weapon_request).Count
        $passed=$request.Count -eq 1 -and $firing.Count -eq 1 -and $stockOK -and $requests -le 3
        $results += [pscustomobject]@{mode=$mode;timescale=$scale;accepted=$passed;requests=$requests;stock=$stock;minimum=($counts|Measure-Object -Minimum).Minimum;trace=$path}
        $results|ConvertTo-Json -Depth 5|Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$results|Format-Table
if (@($results|Where-Object {!$_.accepted}).Count) {throw 'Weapon economy not confirmed; inspect report.json'}
