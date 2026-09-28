[CmdletBinding()]
param([Parameter(Mandatory)][string]$Suite)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Suite).Path
$results = @()
foreach ($run in @(Get-ChildItem -LiteralPath $root -Directory -Filter 'run-*' | Sort-Object Name)) {
    $summary = Get-Content -LiteralPath (Join-Path $run.FullName 'summary.json') -Raw | ConvertFrom-Json
    $rows = @(Get-Content -LiteralPath $summary.trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq 'base3' -and $_.frame -ge 40 })
    if ($rows.Count -lt 150) { throw "$($run.Name): incomplete base3 trace" }
    $search = @($rows | Where-Object { $_.goal -eq 'search_last_seen' -and $_.navigation -eq 'ready' })
    $reacquired = @($rows | Where-Object { $search.Count -gt 0 -and $_.frame -gt $search[0].frame -and $null -ne $_.teammate })
    $guard = @($rows | Where-Object { $_.arbitration.move_limit_reason -eq 'static_laser_hazard' })
    if ($search.Count -eq 0 -or $reacquired.Count -eq 0 -or $guard.Count -lt 5) { throw "$($run.Name): search/reacquisition/laser guard missing" }
    $minHealth = ($rows | Measure-Object -Property health -Minimum).Minimum
    if ($minHealth -lt 100) { throw "$($run.Name): bot took damage (min health $minHealth)" }
    $maxY = ($rows | Where-Object { $_.frame -ge $guard[0].frame } | ForEach-Object { $_.self[1] } | Measure-Object -Maximum).Maximum
    if ($maxY -gt -430) { throw "$($run.Name): bot crossed laser line (max y $maxY)" }
    $results += [pscustomobject]@{run=$run.Name;search_frame=$search[0].frame;reacquired_frame=$reacquired[0].frame;guard_frame=$guard[0].frame;guard_frames=$guard.Count;max_y=$maxY;min_health=$minHealth}
}
if ($results.Count -eq 0) { throw 'No suite runs found' }
$report = Join-Path $root 'laser-corridor-report.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $report -Encoding utf8
$results | Format-Table -AutoSize
Write-Output "Report: $report"
