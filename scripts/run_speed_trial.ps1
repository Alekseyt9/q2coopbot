[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [string]$ServerExe = '',
    [string]$Map = 'base1',
    [int]$Port = 28120,
    [int]$GameFrames = 100,
    [string]$TransitionMap = '',
    [string]$AASDir = '',
    [switch]$RequireTransitionAAS,
    [int]$TransitionAfterFrames = 20,
    [switch]$LeaveTeammateOnTransition,
    [int]$WallLimitSeconds = 0,
    [int[]]$Timescales = @(1, 2),
    [switch]$UnlimitedLoopbackRate,
    [switch]$SynchronizedStart,
    [switch]$ElevatorTrial,
    [switch]$CombatMoveTrial,
    [switch]$ObservationGapTrial,
    [switch]$FriendlyFireTrial,
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
if ($ElevatorTrial -and ($TransitionMap -ne 'base2' -or -not $SynchronizedStart -or -not $AASDir)) {
    throw '-ElevatorTrial requires -TransitionMap base2, -SynchronizedStart and -AASDir.'
}
if ($CombatMoveTrial -and ($Map -ne 'base1' -or $TransitionMap -or -not $SynchronizedStart -or $ElevatorTrial)) {
    throw '-CombatMoveTrial requires -Map base1, -SynchronizedStart, and no map transition or elevator trial.'
}
if ($ObservationGapTrial -and (-not $CombatMoveTrial -or $GameFrames -lt 30)) {
    throw '-ObservationGapTrial requires -CombatMoveTrial and at least 30 game frames.'
}
if ($FriendlyFireTrial -and ($Map -ne 'base1' -or $TransitionMap -or -not $SynchronizedStart -or $ElevatorTrial -or $CombatMoveTrial)) {
    throw '-FriendlyFireTrial requires -Map base1, -SynchronizedStart, and no other gameplay trial.'
}
if ($AASDir -and -not (Test-Path -LiteralPath $AASDir -PathType Container)) { throw "AAS directory is missing: $AASDir" }
if (-not $ServerExe) { $ServerExe = Join-Path $RuntimeRoot 'q2ded.exe' }
$gameDir = Join-Path $RuntimeRoot 'baseq2'
if (-not (Test-Path -LiteralPath $ServerExe) -or -not (Test-Path -LiteralPath $gameDir) -or
    -not (Test-Path -LiteralPath (Join-Path $repoRoot 'go.mod'))) { throw 'Vanilla server runtime or Go module is missing.' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repoRoot ('workspace\artifacts\speed-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$gameDir = [System.IO.Path]::GetFullPath($gameDir)
if ($AASDir) { $AASDir = [System.IO.Path]::GetFullPath($AASDir) }
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
    $humanTracePath = Join-Path $OutputRoot "$name-human-trace.jsonl"
    $botLog = Join-Path $OutputRoot "$name-bot.log"
    $tracePath = Join-Path $OutputRoot "$name-trace.jsonl"
    $worldPath = Join-Path $OutputRoot "$name-world.json"
    $appliedPath = Join-Path $OutputRoot "$name-applied.jsonl"
    $args = "+set game baseq2 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 4 +set port $runPort +set timescale $scale +map $Map"
    $rconPassword = if ($TransitionMap) { [guid]::NewGuid().ToString('N') } else { '' }
    if ($TransitionMap) { $args = "+set rcon_password $rconPassword $args" }
    if ($UnlimitedLoopbackRate) { $args = "+set sv_test_unlimited_loopback 1 $args" }
    if ($ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial) { $args = "+set cheats 1 $args" }
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
        $humanConfigPath = Join-Path $OutputRoot "$name-human-config.json"
        $humanConfig = [ordered]@{
            server = [ordered]@{ host = '127.0.0.1'; port = $runPort }
            client = [ordered]@{ name = 'TestHuman'; game_dir = $gameDir }
            run = [ordered]@{ duration = "$($wallLimit + 10)s"; frame_paced = $true }
            output = [ordered]@{}
            test = [ordered]@{ idle = $true }
        }
        if ($ElevatorTrial) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '10,1408,85'
        }
        if ($CombatMoveTrial -or $FriendlyFireTrial) {
            $humanConfig.test.spawn_map = 'base1'
            $humanConfig.test.spawn_soldier = '96,-200,24'
        }
        if ($FriendlyFireTrial) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.spawn_class = 'monster_infantry'
            $humanConfig.test.line_cross = $true
        }
        if ($LeaveTeammateOnTransition) { $humanConfig.test.exit_on_reconnect = $true }
        $humanConfig | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $humanConfigPath -Encoding UTF8
        $human = Start-Process -FilePath $botExe -ArgumentList "--config `"$humanConfigPath`"" -WorkingDirectory $repoRoot -RedirectStandardOutput $humanLog -RedirectStandardError $humanErr -WindowStyle Hidden -PassThru
        $humanUntil = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $humanUntil) {
            if ($human.HasExited) { throw "Test human exited: $humanErr" }
            if (Select-String -LiteralPath $stdout -Pattern 'TestHuman entered the game' -Quiet) { break }
            Start-Sleep -Milliseconds 100
        }
        if ((Get-Date) -ge $humanUntil) { throw "Test human failed to spawn: $stdout" }
        $botConfigPath = Join-Path $OutputRoot "$name-bot-config.json"
        $botConfig = [ordered]@{
            server = [ordered]@{ host = '127.0.0.1'; port = $runPort }
            client = [ordered]@{ name = 'GoCoopMate'; game_dir = $gameDir }
            run = [ordered]@{ duration = "${wallLimit}s"; frame_paced = $true; game_frames = $GameFrames }
            output = [ordered]@{ world_json = $worldPath; trace_jsonl = $tracePath }
            test = [ordered]@{}
        }
        if ($AASDir) { $botConfig.client.aas_dir = $AASDir }
        if ($TransitionMap) {
            $botConfig.test.change_map = $TransitionMap
            $botConfig.test.change_after_frames = $TransitionAfterFrames
        }
        if ($ElevatorTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '-36,1408,-40'
        }
        if ($FriendlyFireTrial) { $botConfig.test.hold_position = $true }
        if ($ObservationGapTrial) {
            $botConfig.test.observation_gap_start = 1
            $botConfig.test.observation_gap_frames = 12
        }
        $botConfig | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $botConfigPath -Encoding UTF8
        $previousRcon = $env:Q2COOPBOT_TEST_RCON
        try {
            if ($TransitionMap) { $env:Q2COOPBOT_TEST_RCON = $rconPassword }
            & $botExe --config $botConfigPath 2>&1 | Tee-Object -FilePath $botLog | Out-Null
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
        $elevatorStages = [System.Collections.Generic.List[string]]::new()
        $elevatorMoverObserved = $false
        $botNearGate = $false
        $elevatorMinZ = [double]::PositiveInfinity
        $elevatorMaxZ = [double]::NegativeInfinity
        $completedFrame = -1
        $regroupedOnUpperFloor = $false
        $combatMoveFrames = 0
        $combatMoveWithSide = 0
        $staleNeutralFrames = 0
        $attackBeforeGap = 0
        $recoveredActionFrames = 0
        $friendlyBlockedFrames = [System.Collections.Generic.List[int]]::new()
        $friendlyFireFrames = [System.Collections.Generic.List[int]]::new()
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
            if ($ElevatorTrial -and $entry.map -eq 'base2') {
                if ($entry.elevator -and -not $elevatorStages.Contains($entry.elevator)) { $elevatorStages.Add($entry.elevator) }
                if (@($entry.movers | Where-Object { $_.model -eq 50 }).Count -gt 0) { $elevatorMoverObserved = $true }
                if ([math]::Abs($entry.self[0] + 36) -lt 48 -and [math]::Abs($entry.self[1] - 1408) -lt 48 -and [math]::Abs($entry.self[2] + 38) -lt 48) { $botNearGate = $true }
                if ($entry.elevator -in @('board', 'ride', 'exit', 'completed')) {
                    $elevatorMinZ = [math]::Min($elevatorMinZ, [double]$entry.self[2])
                    $elevatorMaxZ = [math]::Max($elevatorMaxZ, [double]$entry.self[2])
                }
                if ($entry.elevator -eq 'completed') { $completedFrame = [int]$entry.frame }
                if ($completedFrame -ge 0 -and $entry.frame -gt $completedFrame -and $null -ne $entry.teammate -and
                    [math]::Abs($entry.self[2] - $entry.teammate[2]) -lt 32 -and
                    [math]::Sqrt([math]::Pow($entry.self[0] - $entry.teammate[0], 2) + [math]::Pow($entry.self[1] - $entry.teammate[1], 2)) -lt 100) {
                    $regroupedOnUpperFloor = $true
                }
            }
            if ($CombatMoveTrial -and $entry.arbitration.aim_source -eq 'enemy' -and
                $entry.arbitration.move_source -in @('route', 'detour') -and ($entry.sent_command.Buttons -band 1)) {
                $combatMoveFrames++
                if ([math]::Abs($entry.sent_command.Side) -gt 10) { $combatMoveWithSide++ }
            }
            if ($ObservationGapTrial) {
                if ($entry.relative_frame -lt 1 -and ($entry.sent_command.Buttons -band 1)) { $attackBeforeGap++ }
                if ($entry.arbitration.limit_reason -eq 'stale_observation' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and
                    $entry.sent_command.Up -eq 0 -and $entry.sent_command.Buttons -eq 0 -and
                    $entry.observation_frame -lt $entry.frame) { $staleNeutralFrames++ }
                if ($entry.relative_frame -ge 13 -and $entry.observation_frame -eq $entry.frame -and
                    ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0 -or $entry.sent_command.Buttons -ne 0)) { $recoveredActionFrames++ }
            }
            if ($FriendlyFireTrial -and @($entry.enemies | Where-Object { $_.clear_shot -eq $true }).Count -gt 0) {
                if ($entry.arbitration.limit_reason -eq 'friendly_line_of_fire' -and $entry.sent_command.Buttons -eq 0) {
                    $friendlyBlockedFrames.Add([int]$entry.frame)
                }
                if ($entry.sent_command.Buttons -band 1) { $friendlyFireFrames.Add([int]$entry.frame) }
            }
        }
        $humanAtTop = $false
        if ($ElevatorTrial -and (Test-Path -LiteralPath $humanTracePath)) {
            foreach ($line in Get-Content -LiteralPath $humanTracePath) {
                $entry = $line | ConvertFrom-Json
                if ($entry.map -eq 'base2' -and [math]::Abs($entry.self[0] - 10) -lt 32 -and [math]::Abs($entry.self[1] - 1408) -lt 32 -and [math]::Abs($entry.self[2] - 24) -lt 32) {
                    $humanAtTop = $true
                    break
                }
            }
        }
        $humanMinY = [double]::PositiveInfinity
        $humanMaxY = [double]::NegativeInfinity
        $humanNearLine = $false
        $humanReturned = $false
        if ($FriendlyFireTrial -and (Test-Path -LiteralPath $humanTracePath)) {
            foreach ($line in Get-Content -LiteralPath $humanTracePath) {
                $entry = $line | ConvertFrom-Json
                if ($entry.map -eq 'base1') {
                    $humanMinY = [math]::Min($humanMinY, [double]$entry.self[1])
                    $humanMaxY = [math]::Max($humanMaxY, [double]$entry.self[1])
                    if ($entry.health -gt 0 -and [math]::Abs($entry.self[0] - 66) -lt 8 -and [math]::Abs($entry.self[1] + 234) -lt 8) { $humanNearLine = $true }
                    if ($humanNearLine -and $entry.health -gt 0 -and [math]::Abs($entry.self[0] - 128) -lt 8 -and [math]::Abs($entry.self[1] + 320) -lt 8) { $humanReturned = $true }
                }
            }
        }
        $fireBeforeBlock = 0
        $fireAfterBlock = 0
        if ($friendlyBlockedFrames.Count -gt 0) {
            foreach ($fireFrame in $friendlyFireFrames) {
                if ($fireFrame -lt $friendlyBlockedFrames[0]) { $fireBeforeBlock++ }
                if ($fireFrame -gt $friendlyBlockedFrames[$friendlyBlockedFrames.Count - 1]) { $fireAfterBlock++ }
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
            elevator_trial = [bool]$ElevatorTrial; elevator_stages = $elevatorStages.ToArray()
            elevator_mover_observed = $elevatorMoverObserved; bot_near_gate = $botNearGate; human_at_top = $humanAtTop
            elevator_min_z = $(if ($ElevatorTrial -and $elevatorMinZ -ne [double]::PositiveInfinity) { $elevatorMinZ } else { $null })
            elevator_max_z = $(if ($ElevatorTrial -and $elevatorMaxZ -ne [double]::NegativeInfinity) { $elevatorMaxZ } else { $null })
            regrouped_on_upper_floor = $regroupedOnUpperFloor
            combat_move_trial = [bool]$CombatMoveTrial; combat_move_frames = $combatMoveFrames; combat_move_with_side = $combatMoveWithSide
            observation_gap_trial = [bool]$ObservationGapTrial; attack_before_gap = $attackBeforeGap
            stale_neutral_frames = $staleNeutralFrames; recovered_action_frames = $recoveredActionFrames
            friendly_fire_trial = [bool]$FriendlyFireTrial; friendly_blocked_frames = $friendlyBlockedFrames.Count
            fire_before_block = $fireBeforeBlock; fire_after_block = $fireAfterBlock
            human_min_y = $(if ($FriendlyFireTrial -and $humanMinY -ne [double]::PositiveInfinity) { $humanMinY } else { $null })
            human_max_y = $(if ($FriendlyFireTrial -and $humanMaxY -ne [double]::NegativeInfinity) { $humanMaxY } else { $null })
            human_near_line = $humanNearLine; human_returned = $humanReturned
            sent_commands = $sent.Count; applied_new_commands = $newCommands.Count
            matched_applied_commands = $matchedSequences.Count; trace_jsonl = $tracePath
            bot_config_json = $botConfigPath; human_config_json = $humanConfigPath
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
if ($ElevatorTrial -and @($results | Where-Object {
    -not $_.bot_near_gate -or -not $_.human_at_top -or -not $_.elevator_mover_observed -or
    -not ($_.elevator_stages -contains 'board') -or -not ($_.elevator_stages -contains 'ride') -or
    -not ($_.elevator_stages -contains 'exit') -or -not ($_.elevator_stages -contains 'completed') -or
    -not ($_.elevator_stages.IndexOf('board') -lt $_.elevator_stages.IndexOf('ride') -and
        $_.elevator_stages.IndexOf('ride') -lt $_.elevator_stages.IndexOf('exit') -and
        $_.elevator_stages.IndexOf('exit') -lt $_.elevator_stages.IndexOf('completed')) -or
    $_.elevator_max_z - $_.elevator_min_z -lt 100 -or -not $_.regrouped_on_upper_floor
}).Count -gt 0) {
    throw "Elevator trial did not complete; inspect positions, mover and stages in $summary"
}
if ($CombatMoveTrial -and @($results | Where-Object { $_.combat_move_frames -le 0 -or $_.combat_move_with_side -le 0 }).Count -gt 0) {
    throw "Combat movement trial did not produce firing while following a route: $summary"
}
if ($ObservationGapTrial -and @($results | Where-Object { $_.attack_before_gap -le 0 -or $_.stale_neutral_frames -le 0 -or $_.recovered_action_frames -le 0 }).Count -gt 0) {
    throw "Observation gap did not stop an active command and recover: $summary"
}
if ($FriendlyFireTrial -and @($results | Where-Object { $_.friendly_blocked_frames -le 0 -or $_.fire_before_block -le 0 -or $_.fire_after_block -le 0 -or -not $_.human_near_line -or -not $_.human_returned }).Count -gt 0) {
    throw "Friendly-fire trial did not show fire, a crossing hold, and resumed fire: $summary"
}
Write-Output "Saved $summary"
