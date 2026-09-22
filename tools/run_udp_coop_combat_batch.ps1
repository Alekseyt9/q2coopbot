[CmdletBinding()]
param(
    [ValidateSet('combat', 'follow', 'retreat', 'phase-retreat', 'rescue', 'cover', 'kill-steal', 'kill-steal-default', 'kill-steal-default-timeline', 'kill-steal-default-melee', 'kill-steal-default-fixture', 'lost-los', 'lost-los-map-fixture', 'elevator', 'elevator-route-probe', 'elevator-natural-route', 'elevator-fixture', 'elevator-player', 'transition')]
    [string]$Scenario = 'combat',
    [int]$FirstSeed = 532,
    [int]$Count = 20,
    [int]$Port = 27952,
    [int]$MoveSeconds = 10,
    [int]$StartupDelayMs = 1800,
    [string]$RuntimeRoot = 'F:\src\quake2\q2coopbot-runtime-bot',
    [string]$RepoRoot = 'F:\src\quake2\q2coopbot-release'
)

$ErrorActionPreference = 'Stop'
$q2ded = Join-Path $RuntimeRoot 'q2ded.exe'
$client = Join-Path $RepoRoot 'tools\q2_client_handshake.py'
$reporter = Join-Path $RepoRoot 'tools\coopbot_event_report.py'
$eventLog = Join-Path $RuntimeRoot 'coopbot_debug_events.jsonl'
$botlibLogSource = Join-Path $RuntimeRoot 'botlib.log'
$artifactRoot = Join-Path $RepoRoot "artifacts\udp-$Scenario-baseline"

if (-not (Test-Path -LiteralPath $q2ded)) {
    throw "q2ded is missing: $q2ded"
}
if (-not (Test-Path -LiteralPath $client)) {
    throw "UDP harness is missing: $client"
}
if (-not (Test-Path -LiteralPath $reporter)) {
    throw "event reporter is missing: $reporter"
}
if (Get-Process q2ded -ErrorAction SilentlyContinue) {
    throw 'q2ded is already running; stop the owned test server before starting a batch'
}

New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
$results = [System.Collections.Generic.List[object]]::new()

$moveForward = 400
$sideSpeed = 0
$yawRate = 0
$attack = $false
$jump = $false
$phaseArgs = @()
$waypointArgs = @()
$harnessDuration = $MoveSeconds + 2
$rescueMode = $false
$roleMode = $false
$killStealMode = $false
$killStealDefaultMode = $false
$killStealDefaultTimelineMode = $false
$killStealDefaultFixtureMode = $false
$elevatorMode = $false
$elevatorRouteMode = $false
$elevatorNaturalRouteMode = $false
$elevatorFixtureMode = $false
$elevatorPlayerMode = $false
$transitionMode = $false
$lostLosMapFixtureMode = $false
switch ($Scenario) {
    'combat' {
        $attack = $true
    }
    'follow' {
        $moveForward = 400
    }
    'retreat' {
        $moveForward = -400
        $sideSpeed = 200
        $yawRate = 35
        $attack = $true
        $jump = $true
    }
    'phase-retreat' {
        $phaseArgs = @(
            '--phase', '4:400:0:0:0:0',
            '--phase', '6:-400:200:35:1:1',
            '--phase', '4:0:0:0:1:0'
        )
        $harnessDuration = 16
    }
    'rescue' {
        $phaseArgs = @(
            '--phase', '3:400:0:0:0:0',
            '--phase', '14:0:0:0:0:0',
            '--phase', '3:-200:0:0:0:0'
        )
        $harnessDuration = 22
        $rescueMode = $true
        $StartupDelayMs = 500
    }
    'cover' {
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '8:-400:200:35:1:1',
            '--phase', '4:0:0:0:1:0'
        )
        $harnessDuration = 18
        $roleMode = $true
    }
    'kill-steal' {
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '8:0:0:0:1:0',
            '--phase', '4:-200:100:25:1:0'
        )
        $harnessDuration = 18
        $killStealMode = $true
    }
    'kill-steal-default' {
        # Natural-map probe using the stock 192-unit yield radius.  The UDP
        # player remains fully live; no position/entity test command is used.
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '8:0:0:0:1:0',
            '--phase', '4:-200:100:25:1:0',
            '--phase', '12:0:0:0:0:0'
        )
        $harnessDuration = 30
        $killStealMode = $true
        $killStealDefaultMode = $true
        $StartupDelayMs = 500
    }
    'kill-steal-default-timeline' {
        # Natural map-aware probe: let the bot acquire the encounter first,
        # then move the live UDP player down the corridor and hold attack.
        # There is no test-mode position/entity override in this scenario.
        $phaseArgs = @(
            '--phase', '4:0:0:0:1:0',
            '--phase', '8:400:0:0:1:0',
            '--phase', '12:0:0:0:1:0'
        )
        $harnessDuration = 26
        $killStealMode = $true
        $killStealDefaultMode = $true
        $killStealDefaultTimelineMode = $true
        $StartupDelayMs = 500
    }
    'kill-steal-default-melee' {
        # Natural route probe toward the known base2 infantry encounter near
        # (-80, 1896, -168).  The player reaches it only through clc_move:
        # advance, turn, advance, then hold attack.
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '2:400:0:45:1:0',
            '--phase', '12:400:0:0:1:0',
            '--phase', '8:0:0:0:1:0'
        )
        $harnessDuration = 28
        $killStealMode = $true
        $killStealDefaultMode = $true
        $StartupDelayMs = 500
    }
    'kill-steal-default-fixture' {
        # Keep real UDP player attack/focus telemetry, while opt-in test
        # commands place the live player and bot at deterministic valid points
        # around a live melee monster beyond the stock 192-unit yield radius.
        $phaseArgs = @(
            '--phase', '4:0:0:0:1:0',
            '--phase', '12:0:0:0:1:0'
        )
        $harnessDuration = 18
        $killStealMode = $true
        $killStealDefaultFixtureMode = $true
        $StartupDelayMs = 500
    }
    'lost-los' {
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '4:0:0:180:1:0',
            '--phase', '8:-400:200:0:0:0',
            '--phase', '12:0:0:0:0:0'
        )
        $harnessDuration = 30
    }
    'lost-los-map-fixture' {
        # Deterministic live-monster LOS fixture.  The UDP player still joins
        # and sends the same real-time phases; only the bot is placed on two
        # known valid base2 points so the target/occluder geometry is fixed.
        $phaseArgs = @('--phase', '40:0:0:0:0:0')
        $harnessDuration = 42
        $lostLosMapFixtureMode = $true
        $StartupDelayMs = 500
    }
    'elevator' {
        $phaseArgs = @(
            '--phase', '4:0:0:56:0:0',
            '--phase', '20:400:0:0:0:0'
        )
        $harnessDuration = 26
        $elevatorMode = $true
        $StartupDelayMs = 500
    }
    'elevator-route-probe' {
        # Natural player route probe: the UDP client advances from the spawn,
        # turns left through real-time yaw usercmds, and holds position.  No
        # test position command is used; map_model only enables diagnostics.
        $phaseArgs = @(
            '--phase', '4:400:0:0:0:0',
            '--phase', '2:400:0:-45:0:0',
            '--phase', '6:400:0:0:0:0',
            '--phase', '6:0:0:0:0:0'
        )
        $harnessDuration = 20
        $elevatorMode = $true
        $elevatorRouteMode = $true
        $StartupDelayMs = 500
    }
    'elevator-natural-route' {
        # AAS-center route from the live base2 spawn to the lower elevator
        # area 806. The client follows it with ordinary UDP usercmds. The
        # optional test state below only guards bot health; no position command
        # is enabled, so player/platform physics remain live.
        $waypointArgs = @(
            '--waypoint', '758.4:2296.0:-232.0',
            '--waypoint', '715.8:2297.1:-232.0',
            '--waypoint', '696.0:2296.0:-232.0',
            '--waypoint', '643.4:2294.5:-232.0',
            '--waypoint', '574.0:2525.7:-232.0',
            '--waypoint', '472.9:2293.8:-232.0',
            '--waypoint', '344.4:2400.0:-232.0',
            '--waypoint', '315.7:2304.0:-232.0',
            '--waypoint', '176.0:2304.0:-221.3',
            '--waypoint', '63.5:2280.0:-168.0',
            '--waypoint', '23.8:2359.9:-168.0',
            '--waypoint=-32.0:2301.8:-168.0',
            '--waypoint=-66.9:2304.0:-168.0',
            '--waypoint=-135.1:2297.3:-168.0',
            '--waypoint=-186.7:2005.7:-168.0',
            '--waypoint=-188.1:1922.8:-168.0',
            '--waypoint=-148.2:1800.1:-168.0',
            '--waypoint=-63.0:1853.2:-168.0',
            '--waypoint=-23.3:1861.0:-168.0',
            '--waypoint', '56.0:1810.7:-168.0',
            '--waypoint', '137.0:1855.8:-168.0',
            '--waypoint', '208.0:1818.6:-168.0',
            '--waypoint', '191.6:1669.2:-168.0',
            '--waypoint', '283.0:1632.0:-168.0',
            '--waypoint', '192.0:1600.0:-168.0',
            '--waypoint', '182.7:1548.3:-168.0',
            '--waypoint', '109.3:1365.7:-168.0',
            '--waypoint', '16.8:1408.0:-168.0',
            '--waypoint=-4.0:1408.0:-168.0',
            # Finish at the center of the lower platform trigger. Once the
            # waypoint is reached, the harness sends zero movement and lets
            # the live func_plat physics lift the human naturally.
            '--waypoint=-32.3:1408.0:-176.0',
            '--waypoint=-84.0:1408.0:-176.0',
            '--waypoint=-56.0:1408.0:-176.0',
            '--waypoint=-84.0:1408.0:-176.0'
        )
        $harnessDuration = 120
        $elevatorMode = $true
        $elevatorNaturalRouteMode = $true
        $StartupDelayMs = 500
    }
    'elevator-fixture' {
        # Opt-in runtime fixture: place the bot at the lower elevator area and
        # the UDP human at the upper area so the live regroup branch must use
        # the detected TRAVEL_ELEVATOR edge.
        $phaseArgs = @('--phase', '35:0:0:0:0:0')
        $harnessDuration = 36
        $elevatorMode = $true
        $elevatorFixtureMode = $true
        $StartupDelayMs = 500
    }
    'elevator-player' {
        # Opt-in real player-platform probe: the UDP human is placed on the
        # lower platform and then held still while the live func_plat moves.
        $phaseArgs = @('--phase', '7:0:0:0:0:0')
        $harnessDuration = 9
        $elevatorMode = $true
        $elevatorPlayerMode = $true
        $StartupDelayMs = 500
    }
    'transition' {
        $phaseArgs = @(
            '--phase', '3:0:0:0:0:0',
            '--phase', '3:400:0:0:0:0'
        )
        $harnessDuration = 16
        $transitionMode = $true
        $StartupDelayMs = 500
    }
}

for ($offset = 0; $offset -lt $Count; $offset++) {
    $seed = $FirstSeed + $offset
    $episode = "coop-$Scenario-$seed"
    $runPort = $Port + $offset
    $serverStdout = Join-Path $artifactRoot "$episode-server.stdout.log"
    $serverStderr = Join-Path $artifactRoot "$episode-server.stderr.log"
    $harnessOutput = Join-Path $artifactRoot "$episode-harness.json"
    $harnessError = Join-Path $artifactRoot "$episode-harness.stderr.log"
    $reportOutput = Join-Path $artifactRoot "$episode-report.json"
    $episodeBotlibLog = Join-Path $artifactRoot "$episode-botlib.log"

    $serverArgs = @(
        '+set', 'game', 'baseq2',
        '+set', 'dedicated', '1',
        '+set', 'coop', '1',
        '+set', 'deathmatch', '0',
        '+set', 'maxclients', '8',
        '+set', 'minimumplayers', '2',
        '+set', 'coopbot_log', '2',
        '+set', 'port', "$runPort",
        '+set', 'botlib', 'libgladiator_x64.dll',
        '+set', 'coopbot_seed', "$seed",
        '+set', 'coopbot_episode_id', $episode,
        '+map', 'base2'
    )
    if ($killStealDefaultFixtureMode) {
        # q2ded has a small argv limit; this opt-in fixture keeps the runtime
        # default game directory instead of spending three argv slots on the
        # explicit `game baseq2` pair.
        $serverArgs = @($serverArgs[3..($serverArgs.Count - 1)])
    }
    if ($rescueMode) {
        $serverArgs = @(
            '+set', 'coopbot_roles', '1',
            '+set', 'coopbot_rescue', '1'
        ) + $serverArgs
    }
    elseif ($roleMode) {
        $serverArgs = @(
            '+set', 'coopbot_roles', '1',
            '+set', 'coopbot_player_intent', '1',
            '+set', 'coopbot_joint_retreat', '1'
        ) + $serverArgs
    }
    elseif ($killStealMode) {
        $killStealRadius = if ($killStealDefaultFixtureMode -or $killStealDefaultMode) { '192' } else { '64' }
        $serverArgs = @(
            '+set', 'coopbot_player_intent', '1',
            '+set', 'coopbot_shared_focus', '1',
            '+set', 'coopbot_kill_steal_control', '1',
            '+set', 'coopbot_kill_steal_radius', $killStealRadius
        ) + $serverArgs
    }
    if ($killStealDefaultFixtureMode) {
        $serverArgs = @('+set', 'coopbot_test_mode', '1') + $serverArgs
    }
    if ($lostLosMapFixtureMode) {
        $serverArgs = @('+set', 'coopbot_test_mode', '1') + $serverArgs
    }
    if ($elevatorFixtureMode -or $elevatorPlayerMode) {
        $serverArgs = @(
            '+set', 'coopbot_test_mode', '1',
            '+set', 'coopbot_map_model', '1'
        ) + $serverArgs
    }
    elseif ($transitionMode) {
        $serverArgs = @(
            '+set', 'coopbot_test_mode', '1'
        ) + $serverArgs
    }
    elseif ($elevatorMode) {
        $serverArgs = @(
            '+set', 'coopbot_map_model', '1'
        ) + $serverArgs
    }
    if ($elevatorRouteMode -or $elevatorNaturalRouteMode) {
        $serverArgs = @(
            '+set', 'coopbot_player_intent', '1'
        ) + $serverArgs
    }
    if ($elevatorNaturalRouteMode) {
        $serverArgs = @('+set', 'coopbot_test_mode', '1') + $serverArgs
    }

    $server = $null
    $harnessExit = 99
    $reportExit = 99
    $report = $null
    try {
        $server = Start-Process -FilePath $q2ded -WorkingDirectory $RuntimeRoot `
            -ArgumentList $serverArgs -RedirectStandardOutput $serverStdout `
            -RedirectStandardError $serverStderr -WindowStyle Hidden -PassThru
        Start-Sleep -Milliseconds $StartupDelayMs

        $harnessArgs = @(
            $client,
            '--port', "$runPort",
            '--duration', ([string]$harnessDuration),
            '--name', "CombatHuman$seed",
            '--event-log', $eventLog,
            '--episode-id', $episode,
            '--require-human-marker'
        )
        $harnessArgs += @('--server-command', 'use blaster')
        if ($transitionMode) {
            $harnessArgs += @(
                '--server-command-at', '4.9:coopbot_test_state 73 50 37',
                '--server-command-at', '5:coopbot_test_changelevel base1'
            )
        }
        if ($elevatorFixtureMode) {
            $harnessArgs += @(
                '--server-command-at', '1:coopbot_test_state 100 100 100',
                '--server-command-at', '1.2:coopbot_test_bot_position -62 1408 -34',
                '--server-command-at', '1.4:coopbot_test_player_position -4 1408 60'
            )
        }
        if ($elevatorPlayerMode) {
            $harnessArgs += @(
                '--server-command-at', '1:coopbot_test_state 100 100 100',
                '--server-command-at', '1.2:coopbot_test_bot_position -120 1408 20',
                '--server-command-at', '1.4:coopbot_test_player_position -60 1408 20'
            )
        }
        if ($killStealDefaultFixtureMode) {
            $harnessArgs += @(
                '--server-command-at', '1.2:coopbot_test_bot_position 160 1896 -168',
                '--server-command-at', '1.4:coopbot_test_player_position -240 1896 -168',
                '--server-command-at', '4.2:coopbot_test_bot_position 160 1896 -168'
            )
            foreach ($positionTime in @(4.6, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8, 8.2, 8.6, 9.0, 9.4, 9.8, 10.2, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8, 14.2, 14.6, 15.0, 15.4, 15.8)) {
                $harnessArgs += @('--server-command-at', "${positionTime}:coopbot_test_bot_position 160 1896 -168")
            }
        }
        if ($lostLosMapFixtureMode) {
            $harnessArgs += @(
                '--server-command-at', '1.2:coopbot_test_bot_position 160 1896 -168',
                '--server-command-at', '24.2:coopbot_test_bot_position 832 2292 -232'
            )
            foreach ($positionTime in @(1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.0, 10.4, 10.8, 11.2, 11.6, 12.0, 12.4, 12.8, 13.2, 13.6, 14.0, 14.4, 14.8, 15.2, 15.6, 16.0, 16.4, 16.8, 17.2, 17.6, 18.0, 18.4, 18.8, 19.2, 19.6, 20.0, 20.4, 20.8, 21.2, 21.6, 22.0, 22.4, 22.8, 23.2, 23.6)) {
                $harnessArgs += @(
                    '--server-command-at', "${positionTime}:coopbot_test_bot_position 160 1896 -168"
                )
            }
        }
        if ($Scenario -eq 'lost-los') {
            # Keep the real UDP player alive long enough to complete the
            # map-aware corridor movement; this sends normal player commands,
            # never a position or entity override.
            $harnessArgs += @(
                '--server-command-at', '0.5:give health 100',
                '--server-command-at', '8:give health 100',
                '--server-command-at', '16:give health 100'
            )
        }
        if ($Scenario -eq 'kill-steal-default') {
            # Avoid an incidental human death ending the natural corridor
            # before the bot can observe and yield the player-focused target.
            $harnessArgs += @(
                '--server-command-at', '0.5:give health 100',
                '--server-command-at', '8:give health 100',
                '--server-command-at', '16:give health 100'
            )
        }
        if ($killStealDefaultTimelineMode) {
            $harnessArgs += @(
                '--server-command-at', '0.5:give health 100',
                '--server-command-at', '8:give health 100',
                '--server-command-at', '16:give health 100'
            )
        }
        if ($rescueMode) {
            $harnessArgs += @('--server-command', 'give health 20')
        }
        if ($Scenario -eq 'elevator-natural-route') {
            # Keep the real UDP player alive while the closed-loop route and
            # elevator regroup finish. These are ordinary server commands;
            # no position/entity override is used.
            $harnessArgs += @(
                '--server-command-at', '0.5:coopbot_test_state 100 100 100',
                '--server-command-at', '0.5:god',
                '--server-command-at', '0.5:notarget',
                '--server-command-at', '0.5:coopbot_test_clear_monster_targets',
                '--server-command-at', '0.5:give health 100',
                '--server-command-at', '16:give health 100',
                '--server-command-at', '32:give health 100',
                '--server-command-at', '48:give health 100',
                '--server-command-at', '64:give health 100',
                '--server-command-at', '80:give health 100'
            )
        }
        if ($waypointArgs.Count -gt 0) {
            $harnessArgs += $waypointArgs
            if ($Scenario -eq 'elevator-natural-route') {
                $harnessArgs += @(
                    '--post-move-duration', '45',
                    '--waypoint-horizontal-tolerance', '16',
                    '--waypoint-vertical-tolerance', '24',
                    '--use'
                )
            }
        }
        elseif ($phaseArgs.Count -gt 0) {
            $harnessArgs += $phaseArgs
        }
        else {
            $harnessArgs += @(
                '--move-forward', "$MoveSeconds",
                '--forward-speed', "$moveForward",
                '--side-speed', "$sideSpeed",
                '--yaw-rate', "$yawRate"
            )
            if ($attack) { $harnessArgs += '--attack' }
            if ($jump) { $harnessArgs += '--jump' }
        }
        & python @harnessArgs 1> $harnessOutput 2> $harnessError
        $harnessExit = $LASTEXITCODE
    }
    finally {
        if ($null -ne $server) {
            $owned = Get-Process -Id $server.Id -ErrorAction SilentlyContinue
            if ($null -ne $owned -and $owned.ProcessName -eq 'q2ded') {
                Stop-Process -Id $server.Id -Force
                Wait-Process -Id $server.Id -Timeout 5 -ErrorAction SilentlyContinue
            }
        }
        if (Test-Path -LiteralPath $botlibLogSource) {
            Copy-Item -LiteralPath $botlibLogSource -Destination $episodeBotlibLog -Force
        }
    }

    $reportArgs = @(
        $reporter, $eventLog, '--episode-id', $episode,
        '--require-human-player', '--output', $reportOutput
    )
    if ($Scenario -eq 'combat') {
        $reportArgs += '--require-bot-monster-damage'
    }
    if (Test-Path -LiteralPath $episodeBotlibLog) {
        $reportArgs += @('--botlib-log', $episodeBotlibLog)
    }
    if ($Scenario -eq 'phase-retreat') {
        $reportArgs += '--require-regroup'
    }
    if ($Scenario -eq 'rescue') {
        $reportArgs += '--require-rescue'
    }
    if ($Scenario -eq 'cover') {
        $reportArgs += @('--require-role', 'COVER')
    }
    if ($Scenario -eq 'kill-steal' -or $Scenario -eq 'kill-steal-default' -or $Scenario -eq 'kill-steal-default-timeline' -or $Scenario -eq 'kill-steal-default-melee' -or $Scenario -eq 'kill-steal-default-fixture') {
        $reportArgs += '--require-kill-steal-yield'
    }
    if ($Scenario -eq 'lost-los' -or $Scenario -eq 'lost-los-map-fixture') {
        $reportArgs += '--require-target-los-lost'
    }
    if ($Scenario -eq 'elevator') {
        $reportArgs += @('--require-elevator-edge', '--require-vertical-elevator-edge')
    }
    if ($Scenario -eq 'elevator-route-probe') {
        $reportArgs += '--require-player-area-transition'
    }
    if ($Scenario -eq 'elevator-natural-route') {
        $reportArgs += @(
            '--require-player-area-transition',
            '--require-elevator-edge',
            '--require-vertical-elevator-edge',
            '--require-elevator-regroup',
            '--require-elevator-reacquired',
            '--require-regroup-complete'
        )
    }
    if ($Scenario -eq 'elevator-fixture') {
        $reportArgs += @(
            '--require-elevator-edge',
            '--require-vertical-elevator-edge',
            '--require-elevator-regroup',
            '--require-elevator-reacquired',
            '--require-regroup-complete'
        )
    }
    if ($Scenario -eq 'elevator-player') {
        $reportArgs += '--require-player-vertical-transition'
    }
    if ($Scenario -eq 'transition') {
        $reportArgs += @(
            '--require-runtime-map-transition',
            '--require-bot-state-persistence'
        )
    }
    & python @reportArgs
    $reportExit = $LASTEXITCODE
    if (Test-Path -LiteralPath $reportOutput) {
        $report = Get-Content -LiteralPath $reportOutput -Raw | ConvertFrom-Json -AsHashtable
    }

    $results.Add([pscustomobject]@{
        episode = $episode
        seed = $seed
        port = $runPort
        harness_exit = $harnessExit
        report_exit = $reportExit
        human_marker_seen = if ($null -ne $report) { $report['telemetry']['coop']['human_player_samples'] -gt 0 } else { $false }
        bot_shots = if ($null -ne $report) { $report['telemetry']['bot_combat']['shots'] } else { 0 }
        bot_monster_damage_events = if ($null -ne $report) { $report['telemetry']['bot_combat']['monster_damage_events'] } else { 0 }
        bot_monster_damage_total = if ($null -ne $report) { $report['telemetry']['bot_combat']['monster_damage_total'] } else { 0 }
        regroup_events = if ($null -ne $report) { $report['telemetry']['coop']['regroup'] } else { 0 }
        stuck_events = if ($null -ne $report) { $report['telemetry']['coop']['stuck'] } else { 0 }
    })

    Write-Host ("{0}: harness={1} report={2} bot_shots={3} bot_monster_damage={4}" -f `
        $episode, $harnessExit, $reportExit, $results[$results.Count - 1].bot_shots,
        $results[$results.Count - 1].bot_monster_damage_events)
}

$summaryPath = Join-Path $artifactRoot "baseline-$FirstSeed-$($FirstSeed + $Count - 1).json"
$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$results | Format-Table -AutoSize
Write-Host "Summary: $summaryPath"

if (Get-Process q2ded -ErrorAction SilentlyContinue) {
    throw 'q2ded is still running after the batch'
}
