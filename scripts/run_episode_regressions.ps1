[CmdletBinding()]
param(
    [string[]]$Id = @(),
    [switch]$List,
    [int[]]$Timescales = @(1,2),
    [int]$Port = 29200
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$registry = Get-Content (Join-Path $repo 'docs/testing/episodes.json') -Raw | ConvertFrom-Json
if ($registry.version -ne 1) { throw 'Unsupported episode registry version' }
$seen = @{}
foreach ($episode in $registry.episodes) {
    if (!$episode.id -or $seen.ContainsKey($episode.id)) { throw 'Missing or duplicate episode ID' }
    $seen[$episode.id] = $true
    if ($episode.reproduction -notin @('ready','observation_only')) { throw "Invalid reproduction: $($episode.id)" }
    if ($episode.reproduction -eq 'ready' -and (!$episode.scenario -or !$episode.acceptance -or $episode.missing_setup.Count)) { throw "Incomplete ready episode: $($episode.id)" }
    if ($episode.scenario -and !(Test-Path (Join-Path $repo $episode.scenario))) { throw "Missing scenario: $($episode.id)" }
}
if ($List) { $registry.episodes | Select-Object id,status,reproduction,title; return }
foreach ($name in $Id) { if (!$seen.ContainsKey($name)) { throw "Unknown episode: $name" } }
$selected = @($registry.episodes | Where-Object { (!$Id.Count -and $_.reproduction -eq 'ready') -or $_.id -in $Id })
foreach ($episode in $selected) { if ($episode.reproduction -ne 'ready') { throw "Episode $($episode.id) cannot run yet: $($episode.missing_setup -join ', ')" } }
if (!$selected.Count -or @($Timescales | Where-Object { $_ -notin @(1,2) }).Count) { throw 'Choose ready episodes and timescales 1/2' }
$out = Join-Path $repo ('workspace/artifacts/episode-regressions-' + (Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out | Out-Null
Copy-Item (Join-Path $repo 'docs/testing/episodes.json') (Join-Path $out 'registry.json')
$exe = Join-Path $out 'q2coopbot.exe'
Push-Location $repo
try {
    & go build -o $exe ./cmd/q2coopbot
    if ($LASTEXITCODE) { throw 'Build failed' }
    $results = @()
    foreach ($episode in $selected) {
        foreach ($scale in $Timescales) {
            $trial = Join-Path $out ($episode.id + '-x' + $scale)
            $passed = $false; $reason = ''
            try {
                $scenario = Join-Path $repo $episode.scenario
                $definition = Get-Content $scenario -Raw | ConvertFrom-Json
                if ($definition.map -ne $episode.map) { throw 'Scenario map mismatch' }
                $args = @{ActorScenario=$scenario;SynchronizedStart=$true;UnlimitedLoopbackRate=$true;GameFrames=$definition.game_frames;Timescales=@($scale);Port=$Port;OutputRoot=$trial;ClientExe=$exe}
                if ($episode.map -ne 'base1') { $args.TransitionMap = $episode.map }
                & (Join-Path $PSScriptRoot 'run_speed_trial.ps1') @args | Out-Host
                $rows = @(Get-Content (Join-Path $trial "scale-$scale-port-$Port-trace.jsonl") | ForEach-Object { ConvertFrom-Json $_ } | Where-Object map -eq $episode.map)
                $accept = $episode.acceptance
                $eventFrame = -1; $landed = $false; $followed = !$accept.follow_after_event
                foreach ($row in $rows) {
                    if ($accept.required_event -and $row.arbitration.limit_reason -eq $accept.required_event -and $row.health -gt 0) { $eventFrame = $row.frame }
                    $inside = $row.health -gt 0 -and $row.on_ground
                    for ($axis=0; $axis -lt 3; $axis++) { $inside = $inside -and $row.self[$axis] -ge $accept.min[$axis] -and $row.self[$axis] -le $accept.max[$axis] }
                    if ($inside -and (!$accept.required_event -or $eventFrame -ge 0)) { $landed = $true }
                    if ($landed -and $row.frame -gt $eventFrame -and $row.health -gt 0 -and $row.goal -eq 'follow_teammate' -and ($row.sent_command.Forward -ne 0 -or $row.sent_command.Side -ne 0)) { $followed = $true }
                }
                $passed = $landed -and $followed
                $reason = if ($passed) {'accepted'} elseif (!$landed) {'landing_or_arrival_not_observed'} else {'follow_not_resumed'}
            } catch { $reason = $_.Exception.Message }
            $results += [pscustomobject]@{id=$episode.id;timescale=$scale;accepted=$passed;reason=$reason;artifacts=$trial}
            $results | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out 'report.json')
        }
    }
    $results | Format-Table id,timescale,accepted,reason
    Write-Output "Saved $out/report.json"
    if (@($results | Where-Object { !$_.accepted }).Count) { throw 'Episode regressions failed; inspect report.json' }
} finally { Pop-Location }
