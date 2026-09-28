[CmdletBinding()]
param([Parameter(Mandatory)][string]$Suite, [switch]$RequireReactivation)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Suite).Path
$results = @()
foreach ($run in @(Get-ChildItem -LiteralPath $root -Directory -Filter 'run-*' | Sort-Object Name)) {
    $summary = Get-Content -LiteralPath (Join-Path $run.FullName 'summary.json') -Raw | ConvertFrom-Json
    $rows = @(Get-Content -LiteralPath $summary.trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq 'base3' -and $_.frame -ge 40 })
    $on = @($rows | Where-Object { @($_.laser_evidence | Where-Object state -eq 'on').Count -eq 2 })
    $pending = @($rows | Where-Object { @($_.laser_evidence | Where-Object reason -eq 'pending_absence').Count -eq 2 })
    $off = @($rows | Where-Object { @($_.laser_evidence | Where-Object state -eq 'off').Count -eq 2 })
    $guard = @($rows | Where-Object { $_.arbitration.move_limit_reason -eq 'static_laser_hazard' })
    if ($on.Count -lt 20 -or $pending.Count -eq 0 -or $off.Count -lt 2 -or $guard.Count -lt 5) {
        throw "$($run.Name): active/guard/complete-frame off transition missing"
    }
    if ($pending[0].frame -le $guard[0].frame -or $off[0].frame -le $pending[0].frame) {
        throw "$($run.Name): laser transition order invalid"
    }
    $cross = @($rows | Where-Object { $_.frame -ge $off[0].frame -and $_.frame -le $off[-1].frame -and $_.self[1] -gt -380 })
    if ($cross.Count -eq 0) { throw "$($run.Name): bot did not cross the laser line during off evidence" }
    $maxYBefore = ($rows | Where-Object { $_.frame -lt $off[0].frame } | ForEach-Object { $_.self[1] } | Measure-Object -Maximum).Maximum
    if ($maxYBefore -gt -430) { throw "$($run.Name): bot crossed before off confirmation" }
    $minHealth = ($rows | Measure-Object -Property health -Minimum).Minimum
    if ($minHealth -lt 100) { throw "$($run.Name): bot took damage (min health $minHealth)" }
    $reactivated = @($rows | Where-Object { $_.frame -gt $off[-1].frame -and @($_.laser_evidence | Where-Object state -eq 'on').Count -eq 2 })
    if ($RequireReactivation -and $reactivated.Count -eq 0) { throw "$($run.Name): beams did not reactivate" }
    $reactivatedFrame = $null
    if ($reactivated.Count -gt 0) { $reactivatedFrame = $reactivated[0].frame }
    $results += [pscustomobject]@{run=$run.Name;pending_frame=$pending[0].frame;off_frame=$off[0].frame;cross_frame=$cross[0].frame;off_last_frame=$off[-1].frame;reactivated_frame=$reactivatedFrame;min_health=$minHealth}
}
if ($results.Count -eq 0) { throw 'No suite runs found' }
$report = Join-Path $root 'laser-off-report.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $report -Encoding utf8
$results | Format-Table -AutoSize
Write-Output "Report: $report"
