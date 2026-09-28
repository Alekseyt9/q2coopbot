[CmdletBinding()]
param([Parameter(Mandatory)][string]$Suite)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Suite).Path
$results = @()
foreach ($run in @(Get-ChildItem -LiteralPath $root -Directory -Filter 'run-*' | Sort-Object Name)) {
    $summary = Get-Content -LiteralPath (Join-Path $run.FullName 'summary.json') -Raw | ConvertFrom-Json
    $rows = @(Get-Content -LiteralPath $summary.trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq 'base3' -and $_.frame -ge 40 })
    $guard = @($rows | Where-Object { $_.arbitration.move_limit_reason -eq 'static_laser_hazard' })
    $returned = @($rows | Where-Object { $null -ne $_.teammate -and $_.teammate[0] -lt -750 -and $_.teammate[1] -lt -500 })
    if ($guard.Count -lt 10 -or $returned.Count -eq 0 -or $returned[0].frame -le $guard[0].frame) {
        throw "$($run.Name): laser wait or player return missing"
    }
    $before = @($rows | Where-Object { $_.frame -ge $guard[0].frame -and $_.frame -lt $returned[0].frame })
    $maxY = ($before | ForEach-Object { $_.self[1] } | Measure-Object -Maximum).Maximum
    if ($maxY -gt -430) { throw "$($run.Name): bot crossed active laser" }
    $resume = @($rows | Where-Object { $_.frame -ge $returned[0].frame -and $_.frame -le $returned[0].frame+8 -and $_.goal -eq 'follow_teammate' -and $_.arbitration.move_source -eq 'route' })
    $cover = @($rows | Where-Object { $_.frame -gt $returned[0].frame -and $_.goal -eq 'cover_teammate' })
    if ($resume.Count -eq 0 -or $cover.Count -eq 0) { throw "$($run.Name): bot did not resume and rejoin player" }
    $minHealth = ($rows | Measure-Object -Property health -Minimum).Minimum
    if ($minHealth -lt 100) { throw "$($run.Name): bot took damage" }
    $results += [pscustomobject]@{run=$run.Name;guard_frame=$guard[0].frame;return_frame=$returned[0].frame;resume_frame=$resume[0].frame;cover_frame=$cover[0].frame;max_y_before_return=$maxY;min_health=$minHealth}
}
if ($results.Count -eq 0) { throw 'No suite runs found' }
$report = Join-Path $root 'laser-player-return-report.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $report -Encoding utf8
$results | Format-Table -AutoSize
Write-Output "Report: $report"
