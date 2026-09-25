[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [string]$ServerExe = '',
    [string]$Map = 'base1',
    [int]$Port = 28120,
    [int]$GameFrames = 100,
    [string]$TransitionMap = '',
    [switch]$RequireTransitionAAS,
    [int]$TransitionAfterFrames = 20,
    [switch]$LeaveTeammateOnTransition,
    [int]$WallLimitSeconds = 0,
    [int[]]$Timescales = @(1, 2),
    [switch]$UnlimitedLoopbackRate,
    [switch]$SynchronizedStart,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($GameFrames -lt 10 -or $Port -lt 1024 -or $Port + $Timescales.Count -gt 65535 -or
    @($Timescales | Where-Object { $_ -lt 1 }).Count -gt 0) { throw 'Invalid frames, timescales or port range.' }
if ($TransitionMap -and ($TransitionMap -notmatch '^[A-Za-z0-9_]+$' -or $TransitionMap -eq $Map -or
    $TransitionAfterFrames -lt 1 -or $TransitionAfterFrames -ge $GameFrames -or -not $SynchronizedStart)) {
    throw 'Map transition requires a distinct simple map name, a frame inside the episode, and -SynchronizedStart.'
}
if ($LeaveTeammateOnTransition -and -not $TransitionMap) { throw '-LeaveTeammateOnTransition requires -TransitionMap.' }
if ($RequireTransitionAAS -and -not $TransitionMap) { throw '-RequireTransitionAAS requires -TransitionMap.' }
if (-not $ServerExe) { $ServerExe = Join-Path $RuntimeRoot 'q2ded.exe' }
$gameDir = Join-Path $RuntimeRoot 'baseq2'
if (-not (Test-Path -LiteralPath $ServerExe) -or -not (Test-Path -LiteralPath $gameDir) -or
    -not (Test-Path -LiteralPath (Join-Path $repoRoot 'go.mod'))) { throw 'Vanilla server runtime or Go module is missing.' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repoRoot ('workspace\artifacts\speed-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$botExe = Join-Path $OutputRoot 'q2coopbot.exe'
Push-Location $repoRoot
try {
    & go build -o $botExe ./cmd/q2coopbot
    if ($LASTEXITCODE -ne 0) { throw 'Go build failed.' }
} finally { Pop-Location }

function Read-AppliedCommands([string]$LogPath) {
    $pattern = 'sv_test_applied_cmd spawncount=(\d+) frame=(\d+) seq=(\d+) kind=(\w+) pitch=(-?\d+) yaw=(-?\d+) roll=(-?\d+) forward=(-?\d+) side=(-?\d+) up=(-?\d+) buttons=(\d+) impulse=(\d+) msec=(\d+) light=(\d+)'
    foreach ($line in Get-Content -LiteralPath $LogPath) {
        if ($line -notmatch $pattern) { continue }
        [pscustomobject]@{
            spawncount = [int]$Matches[1]; frame = [int]$Matches[2]; client_sequence = [uint32]$Matches[3]; kind = $Matches[4]
            command = [ordered]@{
                Pitch = [int]$Matches[5]; Yaw = [int]$Matches[6]; Roll = [int]$Matches[7]
                Forward = [int]$Matches[8]; Side = [int]$Matches[9]; Up = [int]$Matches[10]
                Buttons = [int]$Matches[11]; Impulse = [int]$Matches[12]
                Msec = [int]$Matches[13]; Light = [int]$Matches[14]
            }
        }
    }
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($scale in $Timescales) {
    $index = $results.Count
    $runPort = $Port + $index
    if (Get-NetUDPEndpoint -LocalPort $runPort -ErrorAction SilentlyContinue) { throw "UDP port $runPort is in use." }
    $name = "scale-$scale-port-$runPort"
    $stdout = Join-Path $OutputRoot "$name-server.log"
    $stderr = Join-Path $OutputRoot "$name-server.err.log"
    $humanLog = Join-Path $OutputRoot "$name-human.log"
    $humanErr = Join-Path $OutputRoot "$name-human.err.log"
    $botLog = Join-Path $OutputRoot "$name-bot.log"
    $tracePath = Join-Path $OutputRoot "$name-trace.jsonl"
    $worldPath = Join-Path $OutputRoot "$name-world.json"
    $appliedPath = Join-Path $OutputRoot "$name-applied.jsonl"
    $args = "+set game baseq2 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 4 +set port $runPort +set timescale $scale +map $Map"
    $rconPassword = if ($TransitionMap) { [guid]::NewGuid().ToString('N') } else { '' }
    if ($TransitionMap) { $args = "+set rcon_password $rconPassword $args" }
    if ($UnlimitedLoopbackRate) { $args = "+set sv_test_unlimited_loopback 1 $args" }
    if ($SynchronizedStart) { $args = "+set sv_test_trace_client GoCoopMate +set sv_test_start_client GoCoopMate $args" }
    $server = Start-Process -FilePath $ServerExe -ArgumentList $args -WorkingDirectory $RuntimeRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $human = $null
    try {
        $readyUntil = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $readyUntil) {
            if ($server.HasExited) { throw "Server exited: $stderr" }
            if (Test-Path -LiteralPath $stdout) {
                $ready = -not $SynchronizedStart -or (Select-String -LiteralPath $stdout -Pattern 'sv_test_start_client ready:' -Quiet)
                if ($ready -and (Get-Item -LiteralPath $stdout).Length -gt 0) { break }
            }
            Start-Sleep -Milliseconds 100
        }
        if ((Get-Date) -ge $readyUntil) { throw "Server startup timed out: $stdout" }
        $totalFrames = $GameFrames + $(if ($TransitionMap) { $TransitionAfterFrames + 20 } else { 0 })
        $wallLimit = [int][math]::Ceiling($totalFrames / (10.0 * $scale) * 3 + 20)
        if ($WallLimitSeconds -gt 0) { $wallLimit = $WallLimitSeconds }
        $humanArgs = "--port $runPort --name TestHuman --game-dir `"$gameDir`" --idle --frame-paced --duration $($wallLimit + 10)s"
        if ($LeaveTeammateOnTransition) { $humanArgs += ' --test-exit-on-reconnect' }
        $human = Start-Process -FilePath $botExe -ArgumentList $humanArgs -WorkingDirectory $repoRoot -RedirectStandardOutput $humanLog -RedirectStandardError $humanErr -WindowStyle Hidden -PassThru
        $humanUntil = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $humanUntil) {
            if ($human.HasExited) { throw "Test human exited: $humanErr" }
            if (Select-String -LiteralPath $stdout -Pattern 'TestHuman entered the game' -Quiet) { break }
            Start-Sleep -Milliseconds 100
        }
        if ((Get-Date) -ge $humanUntil) { throw "Test human failed to spawn: $stdout" }
        $botArgs = @('--port', "$runPort", '--name', 'GoCoopMate', '--game-dir', $gameDir, '--world-json', $worldPath, '--trace-jsonl', $tracePath, '--frame-paced', '--game-frames', "$GameFrames", '--duration', "${wallLimit}s")
        if ($TransitionMap) { $botArgs += @('--test-change-map', $TransitionMap, '--test-change-after-frames', "$TransitionAfterFrames") }
        $previousRcon = $env:Q2COOPBOT_TEST_RCON
        try {
            if ($TransitionMap) { $env:Q2COOPBOT_TEST_RCON = $rconPassword }
            & $botExe @botArgs 2>&1 | Tee-Object -FilePath $botLog | Out-Null
        } finally {
            $env:Q2COOPBOT_TEST_RCON = $previousRcon
        }
        if ($LASTEXITCODE -ne 0) { throw "Go bot failed: $botLog" }
        $final = Get-Content -LiteralPath $botLog | Select-String 'finished connected=' | Select-Object -Last 1
        if (-not $final) { throw "Go bot did not finish: $botLog" }
        $fields = @{}
        foreach ($key in @('game_frames', 'frame_gaps', 'server_suppressed', 'game_fps', 'wall_s', 'decode_errors')) {
            if ($final.Line -notmatch "\b$key=([0-9.]+)") { throw "Missing $key in $botLog" }
            $fields[$key] = $Matches[1]
        }
        if ($final.Line -notmatch '\btransition_timeout=(true|false)') { throw "Missing transition_timeout in $botLog" }
        $transitionTimedOut = $Matches[1] -eq 'true'
        $sent = @{}
        $observedMaps = [System.Collections.Generic.List[string]]::new()
        $teammateSeenInTrace = $false
        $teammateSeenAfterTransition = $false
        foreach ($line in Get-Content -LiteralPath $tracePath) {
            $entry = $line | ConvertFrom-Json
            $sent["$($entry.spawncount):$($entry.client_sequence)"] = $entry.sent_command
            if ($null -ne $entry.teammate) { $teammateSeenInTrace = $true }
            if ($TransitionMap -and $entry.map -eq $TransitionMap -and $null -ne $entry.teammate) {
                $teammateSeenAfterTransition = $true
            }
            if ($entry.map -and ($observedMaps.Count -eq 0 -or $observedMaps[$observedMaps.Count - 1] -cne $entry.map)) {
                $observedMaps.Add($entry.map)
            }
        }
        $applied = @(Read-AppliedCommands $stdout)
        foreach ($item in $applied) { ConvertTo-Json -InputObject $item -Compress -Depth 4 | Add-Content -LiteralPath $appliedPath -Encoding UTF8 }
        $newCommands = @($applied | Where-Object { $_.kind -eq 'new' })
        $matchedSequences = @{}
        foreach ($item in $newCommands) {
            $key = "$($item.spawncount):$($item.client_sequence)"
            if ($sent.ContainsKey($key) -and
                (ConvertTo-Json -InputObject $sent[$key] -Compress -Depth 3) -ceq
                (ConvertTo-Json -InputObject $item.command -Compress -Depth 3)) {
                $matchedSequences[$key] = $true
            }
        }
        $world = Get-Content -LiteralPath $worldPath -Raw | ConvertFrom-Json
        $result = [pscustomobject]@{
            timescale = $scale; map = $Map; final_map = $world.map; observed_maps = $observedMaps.ToArray()
            transition_requested = [bool]$TransitionMap; transition_observed = $TransitionMap -and $observedMaps.Count -ge 2 -and $observedMaps[0] -eq $Map -and $observedMaps[$observedMaps.Count - 1] -eq $TransitionMap
            transition_timeout = $transitionTimedOut
            game_frames = [int]$fields.game_frames
            frame_gaps = [int]$fields.frame_gaps; server_suppressed = [int]$fields.server_suppressed
            game_fps = [double]::Parse($fields.game_fps, [cultureinfo]::InvariantCulture)
            wall_seconds = [double]::Parse($fields.wall_s, [cultureinfo]::InvariantCulture)
            decode_errors = [int]$fields.decode_errors; navigation = $world.navigation
            aas_loaded = [bool]$world.aas_loaded; aas_areas = [int]$world.areas; aas_reachabilities = [int]$world.reachabilities
            teammate_seen = $teammateSeenInTrace
            teammate_seen_after_transition = $teammateSeenAfterTransition
            teammate_left_on_transition = [bool]$LeaveTeammateOnTransition
            sent_commands = $sent.Count; applied_new_commands = $newCommands.Count
            matched_applied_commands = $matchedSequences.Count; trace_jsonl = $tracePath
            applied_jsonl = $appliedPath; world_json = $worldPath; server_log = $stdout
        }
        $results.Add($result)
        $result | Format-Table timescale, game_frames, game_fps, wall_seconds, teammate_seen, matched_applied_commands -AutoSize
    } finally {
        if ($null -ne $human -and -not $human.HasExited) { Stop-Process -Id $human.Id -Force }
        if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    }
}

$summary = Join-Path $OutputRoot 'summary.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summary -Encoding UTF8
if ($TransitionMap -and @($results | Where-Object { $_.transition_timeout -or -not $_.transition_observed -or $_.final_map -ne $TransitionMap }).Count -gt 0) {
    throw "Map transition was not observed; inspect transition_timeout, observed_maps and server log in $summary"
}
if ($TransitionMap -and -not $LeaveTeammateOnTransition -and
    @($results | Where-Object { -not $_.teammate_seen_after_transition }).Count -gt 0) {
    throw "Teammate was not observed after map transition: $summary"
}
if ($RequireTransitionAAS -and @($results | Where-Object {
    -not $_.aas_loaded -or $_.aas_areas -le 0 -or $_.aas_reachabilities -le 0 -or $_.navigation -ne 'ready'
}).Count -gt 0) {
    throw "Transition map AAS route was not ready: $summary"
}
if (@($results | Where-Object { $_.game_frames -lt $GameFrames -or $_.frame_gaps -gt 0 -or $_.decode_errors -gt 0 -or -not $_.teammate_seen }).Count -gt 0) {
    throw "Speed trial failed observation gate: $summary"
}
if ($SynchronizedStart -and @($results | Where-Object { $_.matched_applied_commands -ne $_.sent_commands -or $_.applied_new_commands -ne $_.sent_commands }).Count -gt 0) {
    throw "Server did not apply every sent command: $summary"
}
Write-Output "Saved $summary"
