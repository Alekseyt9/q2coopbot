[CmdletBinding()]
param(
    [ValidateSet('combat', 'follow', 'retreat')]
    [string]$Scenario = 'combat',
    [int]$FirstSeed = 532,
    [int]$Count = 20,
    [int]$Port = 27952,
    [int]$MoveSeconds = 10,
    [string]$RuntimeRoot = 'F:\src\quake2\q2coopbot-runtime-bot',
    [string]$RepoRoot = 'F:\src\quake2\q2coopbot-release'
)

$ErrorActionPreference = 'Stop'
$q2ded = Join-Path $RuntimeRoot 'q2ded.exe'
$client = Join-Path $RepoRoot 'tools\q2_client_handshake.py'
$reporter = Join-Path $RepoRoot 'tools\coopbot_event_report.py'
$eventLog = Join-Path $RuntimeRoot 'coopbot_debug_events.jsonl'
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

    $serverArgs = @(
        '+set', 'game', 'baseq2',
        '+set', 'dedicated', '1',
        '+set', 'coop', '1',
        '+set', 'deathmatch', '0',
        '+set', 'maxclients', '8',
        '+set', 'minimumplayers', '2',
        '+set', 'coopbot_log', '2',
        '+set', 'port', "$runPort",
        '+set', 'logfile', '2',
        '+set', 'botlib', 'libgladiator_x64.dll',
        '+set', 'coopbot_seed', "$seed",
        '+set', 'coopbot_episode_id', $episode,
        '+map', 'base2'
    )

    $server = $null
    $harnessExit = 99
    $reportExit = 99
    $report = $null
    try {
        $server = Start-Process -FilePath $q2ded -WorkingDirectory $RuntimeRoot `
            -ArgumentList $serverArgs -RedirectStandardOutput $serverStdout `
            -RedirectStandardError $serverStderr -WindowStyle Hidden -PassThru
        Start-Sleep -Milliseconds 1800

        $harnessArgs = @(
            $client,
            '--port', "$runPort",
            '--duration', ([string]($MoveSeconds + 2)),
            '--name', "CombatHuman$seed",
            '--event-log', $eventLog,
            '--episode-id', $episode,
            '--require-human-marker',
            '--server-command', 'use blaster',
            '--move-forward', "$MoveSeconds",
            '--forward-speed', "$moveForward",
            '--side-speed', "$sideSpeed",
            '--yaw-rate', "$yawRate"
        )
        if ($attack) { $harnessArgs += '--attack' }
        if ($jump) { $harnessArgs += '--jump' }
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
    }

    $reportArgs = @(
        $reporter, $eventLog, '--episode-id', $episode,
        '--require-human-player', '--output', $reportOutput
    )
    if ($Scenario -eq 'combat') {
        $reportArgs += '--require-bot-monster-damage'
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
