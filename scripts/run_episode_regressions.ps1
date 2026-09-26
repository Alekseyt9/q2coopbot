[CmdletBinding()]
param(
    [string[]]$Id = @(),
    [switch]$List,
    [int[]]$Timescales = @(1,2),
    [int]$Port = 29200
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'check_episode_setup.ps1')
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'read_episode_registry.ps1')
$registry = Read-EpisodeRegistry (Join-Path $repo 'scripts/scenarios/episodes/index.json')
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
$registry | ConvertTo-Json -Depth 40 | Set-Content (Join-Path $out 'registry.json')
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
				Assert-EpisodeSetup $rows $accept
				$stageFrame = -1
				if ($accept.required_move_reason) {
					$moveEvent=@($rows|Where-Object { $_.arbitration.move_limit_reason -eq $accept.required_move_reason -and $_.health -gt 0 -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0) }|Select-Object -First 1)
					if (!$moveEvent.Count) {throw 'Required guarded movement not observed'}
					$stageFrame=$moveEvent[0].frame
				}
				if ($accept.health_after_damage) {
					if (@($rows | Where-Object test_health_masked).Count) { throw 'Natural health test used a perception mask' }
					$initial=@($rows|Where-Object { $_.health -eq 100 -and @($_.pickups|Where-Object class -eq 'item_health').Count }|Select-Object -First 1)
					$actorRows=@(Get-Content (Join-Path $trial "scale-$scale-port-$Port-human-trace.jsonl")|ConvertFrom-Json)
					$shot=@($actorRows|Where-Object { $_.scenario.step_id -eq 'server-damage' -and ($_.sent_command.Buttons -band 1) }|Select-Object -First 1)
					if (!$initial.Count -or !$shot.Count) { throw 'Initial health observation or actor shot missing' }
					$lost=@($rows|Where-Object { $_.frame -gt $initial[0].frame -and $_.frame -lt $shot[0].frame -and $_.health -eq 100 -and @($_.resource_memory|Where-Object { $_.item.class -eq 'item_health' -and $_.state -eq 'unknown' }).Count }|Select-Object -First 1)
					$hurt=@($rows|Where-Object { $_.frame -gt $shot[0].frame -and $_.health -gt 0 -and $_.health -lt 45 }|Select-Object -First 1)
					if (!$lost.Count -or !$hurt.Count) { throw 'Leaving health while healthy or later damage missing' }
					$healed=$null
					for ($i=1;$i -lt $rows.Count;$i++) {
						$row=$rows[$i]; $prev=$rows[$i-1]
						if ($row.frame -le $hurt[0].frame -or $row.health -lt $accept.min_health -or $row.health -le $prev.health -or $prev.goal -ne 'recover_health') { continue }
						foreach ($item in $initial[0].pickups | Where-Object class -eq 'item_health') {
							if ([math]::Abs($row.self[0]-$item.origin[0]) -lt 48 -and [math]::Abs($row.self[1]-$item.origin[1]) -lt 48 -and [math]::Abs($row.self[2]-$item.origin[2]-9.125) -lt 32) { $healed=$row; break }
						}
						if ($healed) {break}
					}
					if (!$healed) { throw 'Treatment at previously observed health not confirmed after damage' }
					$stageFrame=$healed.frame
				}
				if ($accept.memory_health_return) {
					$choice = @($rows | Where-Object { $_.test_health_masked -and $_.goal -eq 'recover_health' -and !@($_.pickups | Where-Object class -eq 'item_health').Count -and @($_.resource_memory | Where-Object { $_.item.class -eq 'item_health' -and $_.state -eq 'unknown' -and $_.attempted }).Count } | Select-Object -First 1)
					if (!$choice.Count) { throw 'Health goal from memory not observed' }
					$target = $choice[0].goal_point
					$remembered = @($choice[0].resource_memory | Where-Object { $_.item.class -eq 'item_health' -and [math]::Abs($_.item.origin[0]-$target[0]) -lt 1 -and [math]::Abs($_.item.origin[1]-$target[1]) -lt 1 } | Select-Object -First 1)
					if (!$remembered.Count) { throw 'Health goal differs from memory target' }
					$entity = $remembered[0].item.id
					if (!@($rows | Where-Object { $_.frame -lt $choice[0].frame -and @($_.pickups | Where-Object id -eq $entity).Count }).Count) { throw 'Health target never observed before memory choice' }
					$healed = @($rows | Where-Object { $_.test_health_masked -and $_.frame -gt $choice[0].frame -and $_.health -ge $accept.min_health -and !@($_.pickups | Where-Object class -eq 'item_health').Count -and [math]::Abs($_.self[0]-$target[0]) -lt 48 -and [math]::Abs($_.self[1]-$target[1]) -lt 48 -and [math]::Abs($_.self[2]-$target[2]) -lt 32 } | Select-Object -First 1)
					if (!$healed.Count -or $healed[0].health -le $choice[0].health) { throw 'Server healing at remembered target not observed' }
					$stageFrame = $healed[0].frame
				}
				if ($accept.memory_missing) {
					$class = $accept.memory_missing
					$observed = @($rows | Where-Object { @($_.resource_memory | Where-Object { $_.item.class -eq $class -and $_.state -eq 'observed' }).Count } | Select-Object -First 1)
					$approach = @($rows | Where-Object { $_.pickup.class -eq $class -and $_.pickup.from_memory -and $_.pickup.state -eq 'approach' -and $_.goal -eq 'collect_item' } | Select-Object -First 1)
					if (!$observed.Count -or !$approach.Count -or $observed[0].frame -ge $approach[0].frame) { throw 'Memory observation/approach missing' }
					$entity = $approach[0].pickup.entity
					$missing = @($rows | Where-Object { $_.frame -gt $approach[0].frame -and @($_.resource_memory | Where-Object { $_.item.id -eq $entity -and $_.state -eq 'unavailable' }).Count } | Select-Object -First 1)
					if (!$missing.Count) { throw 'Missing resource not checked on arrival' }
					if (@($rows | Where-Object { $_.pickup.entity -eq $entity -and ($_.pickup.state -eq 'confirmed' -or $_.pickup.started_frame -gt $missing[0].frame) }).Count) { throw 'False pickup confirmation or repeated missing-resource attempt' }
					$actorRows = @(Get-Content (Join-Path $trial "scale-$scale-port-$Port-human-trace.jsonl") | ConvertFrom-Json)
					$before = @($actorRows | Where-Object { $_.frame -le $observed[0].frame } | Select-Object -Last 1)
					if (!$before.Count -or !$accept.actor_armor_gain -or !@($actorRows | Where-Object { $_.frame -gt $before[0].frame -and $_.frame -lt $approach[0].frame -and $_.armor -ge ($before[0].armor + $accept.actor_armor_gain) }).Count) { throw 'Actor armor pickup not confirmed' }
					$stageFrame = $missing[0].frame
					if ($rows[-1].frame - $stageFrame -lt 30) { throw 'Insufficient observation after missing-resource check' }
				}
				foreach ($class in $accept.inventory_pickups) {
					$confirmed = @($rows | Where-Object {
						$_.pickup.class -eq $class -and $_.pickup.state -eq 'confirmed' -and
						$_.frame -eq $_.pickup.end_frame -and $_.inventory_known -and $_.inventory_age_frames -le 20 -and
						$_.pickup.after -gt $_.pickup.before -and
						@($_.inventory | Where-Object name -eq $_.pickup.name).Count -gt 0
					} | Select-Object -First 1)
					if (!$confirmed.Count) { throw "Inventory pickup not confirmed: $class" }
					$event = $confirmed[0]
					$approach = @($rows | Where-Object {
						$_.pickup.entity -eq $event.pickup.entity -and $_.pickup.started_frame -eq $event.pickup.started_frame -and
						$_.pickup.state -eq 'approach' -and $_.goal -eq 'collect_item' -and $_.frame -lt $event.frame
					})
					if (!$approach.Count) { throw "Autonomous pickup approach not observed: $class" }
					$stageFrame = [math]::Max($stageFrame, $event.frame)
				}
				foreach ($stage in $accept.required_elevator_stages) {
					$match = @($rows | Where-Object { $_.frame -gt $stageFrame -and $_.elevator -eq $stage } | Select-Object -First 1)
					if (!$match.Count) { throw "Elevator stage not observed in order: $stage" }
					$stageFrame = $match[0].frame
				}
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
                    $inside = $row.health -gt 0 -and $row.on_ground -and $row.frame -ge $stageFrame
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
