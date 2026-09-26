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
	if ($episode.asset) { if ((Get-FileHash (Join-Path $repo $episode.asset.path)).Hash -ne $episode.asset.sha256) {throw "Asset hash mismatch: $($episode.id)"} }
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
	$reporter=Join-Path $out 'q2scenario-report.exe'
	& go build -o $reporter ./cmd/q2scenario-report
	if ($LASTEXITCODE) {throw 'Reporter build failed'}
    $results = @()
    foreach ($episode in $selected) {
        foreach ($scale in $Timescales) {
            $trial = Join-Path $out ($episode.id + '-x' + $scale)
            $passed = $false; $reason = ''
            try {
                $scenario = Join-Path $repo $episode.scenario
                $definition = $null
                if ($episode.kind -ne 'weapon_switch') { $definition = Get-Content $scenario -Raw | ConvertFrom-Json }
				if ($episode.kind -eq 'weapon_switch') {
					& $scenario -Timescales @($scale) -Port $Port -OutputRoot $trial | Out-Host
					$report = @(Get-Content (Join-Path $trial 'report.json') -Raw | ConvertFrom-Json)
					if ($report.Count -ne 2 -or @($report | Where-Object { !$_.accepted }).Count) { throw 'Weapon switch trial rejected' }
					$passed=$true; $reason='accepted'
				} elseif ($episode.kind -eq 'elevator') {
					& (Join-Path $PSScriptRoot 'run_speed_trial.ps1') -ElevatorTrial -TransitionMap $definition.map -AASDir (Join-Path $repo $definition.aas_dir) -SynchronizedStart -UnlimitedLoopbackRate -GameFrames $definition.game_frames -Timescales @($scale) -Port $Port -OutputRoot $trial | Out-Host
					$summary = Get-Content (Join-Path $trial 'summary.json') -Raw | ConvertFrom-Json
					if (!$summary.regrouped_on_upper_floor -or 'completed' -notin $summary.elevator_stages) { throw 'Elevator trial rejected' }
					$passed=$true; $reason='accepted'
				} elseif ($episode.kind -eq 'session') {
					New-Item -ItemType Directory $trial|Out-Null
					$trialConfig=Join-Path $trial 'trial.json'
					@{session=$scenario;runtime_root=(Join-Path $repo 'workspace/runtime/q2go');port=$Port;timescale=$scale;tail_frames=5}|ConvertTo-Json|Set-Content $trialConfig
					& (Join-Path $PSScriptRoot 'run_session_trial.ps1') -Config $trialConfig -PreparedOutput $trial -ClientExe $exe -ReporterExe $reporter|Out-Host
					$report=Get-Content (Join-Path $trial 'report.json') -Raw|ConvertFrom-Json
					if (!$report.accepted) {throw 'Session rejected'}
					$passed=$true;$reason='accepted'
				} else {
                if ($definition.map -ne $episode.map) { throw 'Scenario map mismatch' }
                $args = @{ActorScenario=$scenario;SynchronizedStart=$true;UnlimitedLoopbackRate=$true;GameFrames=$definition.game_frames;Timescales=@($scale);Port=$Port;OutputRoot=$trial;ClientExe=$exe}
                if ($episode.map -ne 'base1') { $args.TransitionMap = $episode.map }
                & (Join-Path $PSScriptRoot 'run_speed_trial.ps1') @args | Out-Host
                $rows = @(Get-Content (Join-Path $trial "scale-$scale-port-$Port-trace.jsonl") | ForEach-Object { ConvertFrom-Json $_ } | Where-Object map -eq $episode.map)
                $accept = $episode.acceptance
				if ($accept.actor_health_checkpoints) {
					$actorRows = @(Get-Content (Join-Path $trial "scale-$scale-port-$Port-human-trace.jsonl") | ForEach-Object { ConvertFrom-Json $_ } | Where-Object map -eq $episode.map)
					$after = -1
					foreach ($checkpoint in $accept.actor_health_checkpoints) {
						$match = @($actorRows | Where-Object {
							$_.frame -gt $after -and $_.health -eq $checkpoint.health -and
							[math]::Abs($_.self[0]-$checkpoint.origin[0]) -lt 16 -and
							[math]::Abs($_.self[1]-$checkpoint.origin[1]) -lt 16 -and
							[math]::Abs($_.self[2]-$checkpoint.origin[2]) -lt 16
						} | Select-Object -First 1)
						if (!$match.Count) { throw 'Actor health/position checkpoint not observed' }
						$after = $match[0].frame
					}
				}
				if ($definition.bot_health) {
					$setup=@($rows|Where-Object { $_.health -eq $definition.bot_health -and [math]::Abs($_.self[0]-$definition.bot_origin[0]) -lt 16 -and [math]::Abs($_.self[1]-$definition.bot_origin[1]) -lt 16 })
					if (!$setup.Count) {throw 'Initial bot health/position not observed'}
				}
                $eventFrame = -1; $landed = $false; $followed = !$accept.follow_after_event
                foreach ($row in $rows) {
                    if ($accept.required_event -and $row.arbitration.limit_reason -eq $accept.required_event -and $row.health -gt 0) { $eventFrame = $row.frame }
                    $inside = $row.health -gt 0 -and $row.on_ground
					if ($accept.min_health) {$inside=$inside -and $row.health -ge $accept.min_health}
					if ($accept.goal) {$inside=$inside -and $row.goal -eq $accept.goal}
                    for ($axis=0; $axis -lt 3; $axis++) { $inside = $inside -and $row.self[$axis] -ge $accept.min[$axis] -and $row.self[$axis] -le $accept.max[$axis] }
                    if ($inside -and (!$accept.required_event -or $eventFrame -ge 0)) { $landed = $true }
                    if ($landed -and $row.frame -gt $eventFrame -and $row.health -gt 0 -and $row.goal -eq 'follow_teammate' -and ($row.sent_command.Forward -ne 0 -or $row.sent_command.Side -ne 0)) { $followed = $true }
                }
                $passed = $landed -and $followed
                $reason = if ($passed) {'accepted'} elseif (!$landed) {'landing_or_arrival_not_observed'} else {'follow_not_resumed'}
				}
            } catch { $reason = $_.Exception.Message }
            $results += [pscustomobject]@{id=$episode.id;timescale=$scale;accepted=$passed;reason=$reason;artifacts=$trial}
            $results | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $out 'report.json')
        }
    }
    $results | Format-Table id,timescale,accepted,reason
    Write-Output "Saved $out/report.json"
    if (@($results | Where-Object { !$_.accepted }).Count) { throw 'Episode regressions failed; inspect report.json' }
} finally { Pop-Location }
