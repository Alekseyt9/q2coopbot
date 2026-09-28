[CmdletBinding()]
param([Parameter(Mandatory)][string]$Suite)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Suite).Path
$results = @()
foreach ($run in @(Get-ChildItem -LiteralPath $root -Directory -Filter 'run-*' | Sort-Object Name)) {
    $summary = Get-Content -LiteralPath (Join-Path $run.FullName 'summary.json') -Raw | ConvertFrom-Json
    $rows = @(Get-Content -LiteralPath $summary.trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq 'base3' -and $_.frame -ge 40 })
    $lost = @($rows | Where-Object { $null -eq $_.teammate -and $_.teammate_age_frames -gt 0 } | Select-Object -First 1)
    if ($lost.Count -ne 1) { throw "$($run.Name): no hidden-teammate interval" }
    $search = @($rows | Where-Object { $_.frame -ge $lost[0].frame -and $_.goal -eq 'search_last_seen' -and $_.navigation -eq 'ready' -and $_.search_route.reason -eq 'ready' -and $_.search_route.limit_horizontal_units -eq 1400 })
    if ($search.Count -eq 0) { throw "$($run.Name): no bounded long-range search" }
    $regained = @($rows | Where-Object { $_.frame -gt $search[0].frame -and $null -ne $_.teammate } | Select-Object -First 1)
    if ($regained.Count -ne 1) { throw "$($run.Name): teammate was not reacquired" }
    $covered = @($rows | Where-Object { $_.frame -ge $regained[0].frame -and $_.goal -eq 'cover_teammate' -and $null -ne $_.teammate } | Select-Object -First 1)
    $laser = @($rows | Where-Object { $_.frame -gt $regained[0].frame -and $_.arbitration.move_limit_reason -eq 'static_laser_hazard' } | Select-Object -First 1)
    if ($covered.Count -ne 1 -and $laser.Count -ne 1) { throw "$($run.Name): neither covering range nor an explicit laser block" }
    $outcome = if ($covered.Count -eq 1) { 'cover_teammate' } else { 'blocked_by_laser' }
    $outcomeFrame = if ($covered.Count -eq 1) { $covered[0].frame } else { $laser[0].frame }
    $results += [pscustomobject]@{ run=$run.Name; lost_frame=$lost[0].frame; search_frame=$search[0].frame; reacquired_frame=$regained[0].frame; outcome=$outcome; outcome_frame=$outcomeFrame }
}
if ($results.Count -eq 0) { throw 'No suite runs found' }
$report = Join-Path $root 'lost-follow-report.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $report -Encoding utf8
$results | Format-Table -AutoSize
Write-Output "Report: $report"
