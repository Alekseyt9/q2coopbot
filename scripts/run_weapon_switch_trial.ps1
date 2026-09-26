[CmdletBinding()]
param([int[]]$Timescales = @(2), [int]$Port = 29410, [string]$OutputRoot = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (!$OutputRoot) { $OutputRoot = Join-Path $repo ('workspace/artifacts/weapon-switch-' + (Get-Date -Format yyyyMMdd-HHmmss)) }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$results = @()
foreach ($mode in @('blaster','stocked')) {
    $out = Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -CombatMoveTrial -WeaponSwitchTrial $mode -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 120 -Timescales $Timescales -Port $Port -OutputRoot $out
    foreach ($scale in $Timescales) {
        $path = @(Get-ChildItem $out -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')[0].FullName
        $rows = @(Get-Content $path | ForEach-Object { ConvertFrom-Json $_ })
        $loaded = @($rows | Where-Object { $_.weapon -like '*/v_shotg/*' -and $_.ammo -eq 1 -and ($_.sent_command.Buttons -band 1) } | Select-Object -First 1)
        $empty = @($rows | Where-Object { $loaded.Count -and $_.frame -gt $loaded[0].frame -and $_.weapon -like '*/v_shotg/*' -and $_.ammo -eq 0 } | Select-Object -First 1)
        $expected = if ($mode -eq 'blaster') {'use Blaster'} else {'use Machinegun'}
        $request = @($rows | Where-Object { $empty.Count -and $_.frame -ge $empty[0].frame -and $_.weapon_request -eq $expected -and $_.inventory_known -and $_.inventory_age_frames -le 20 } | Select-Object -First 1)
        $firing = @($rows | Where-Object {
            $request.Count -and $_.frame -gt $request[0].frame -and ($_.sent_command.Buttons -band 1) -and
            (($mode -eq 'blaster' -and $_.weapon -eq 'Blaster') -or ($mode -eq 'stocked' -and $_.weapon -like '*/v_machn/*' -and $_.ammo -gt 0))
        } | Select-Object -First 1)
        $requests = @($rows | Where-Object weapon_request).Count
        $passed = $loaded.Count -eq 1 -and $empty.Count -eq 1 -and $request.Count -eq 1 -and $firing.Count -eq 1 -and $requests -le 3
        $results += [pscustomobject]@{mode=$mode;timescale=$scale;accepted=$passed;requests=$requests;trace=$path}
        $results | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$results | Format-Table
if (@($results | Where-Object { !$_.accepted }).Count) { throw "Weapon switching was not confirmed: $OutputRoot/report.json" }
