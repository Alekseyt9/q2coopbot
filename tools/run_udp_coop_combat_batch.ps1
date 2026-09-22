[CmdletBinding()]
param(
    [ValidateSet('combat', 'follow', 'retreat', 'phase-retreat', 'rescue', 'cover', 'kill-steal', 'lost-los', 'transition')]
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
$harnessDuration = $MoveSeconds + 2
$rescueMode = $false
$roleMode = $false
$killStealMode = $false
$transitionMode = $false
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
    'lost-los' {
        $phaseArgs = @(
            '--phase', '4:400:0:0:1:0',
            '--phase', '4:0:0:180:1:0',
            '--phase', '8:-400:200:0:0:0'
        )
        $harnessDuration = 18
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
        # Controlled probe: make the yield threshold observable on base2;
        # the game default remains 192 and is covered by the separate probe.
        $serverArgs = @(
            '+set', 'coopbot_player_intent', '1',
            '+set', 'coopbot_shared_focus', '1',
            '+set', 'coopbot_kill_steal_control', '1',
            '+set', 'coopbot_kill_steal_radius', '64'
        ) + $serverArgs
    }
    elseif ($transitionMode) {
        $serverArgs = @(
            '+set', 'coopbot_test_mode', '1'
        ) + $serverArgs
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
        if ($rescueMode) {
            $harnessArgs += @('--server-command', 'give health 20')
        }
        if ($phaseArgs.Count -gt 0) {
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
    if ($Scenario -eq 'kill-steal') {
        $reportArgs += '--require-kill-steal-yield'
    }
    if ($Scenario -eq 'lost-los') {
        $reportArgs += '--require-target-lost'
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
