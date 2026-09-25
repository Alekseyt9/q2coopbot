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
    [switch]$GroundEdgeTrial,
    [switch]$NoAASTrial,
    [switch]$DoorTrial,
    [switch]$HideDoorMover,
    [switch]$DoorPassTrial,
    [switch]$ButtonTrial,
    [switch]$ButtonAutoTrial,
    [switch]$TeammateMemoryTrial,
    [switch]$TeammateSearchTrial,
    [string]$BSPFailureTrial = '',
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
if ($GroundEdgeTrial -and ($Map -ne 'base1' -or $TransitionMap -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial)) {
    throw '-GroundEdgeTrial requires base1, synchronized start, and no other gameplay trial.'
}
if ($NoAASTrial -and ($Map -ne 'base1' -or $TransitionMap -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial)) {
    throw '-NoAASTrial requires base1, synchronized start, and no other gameplay trial except -GroundEdgeTrial.'
}
if ($DoorTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or $GroundEdgeTrial -or $NoAASTrial)) {
    throw '-DoorTrial requires synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($HideDoorMover -and -not $DoorTrial) { throw '-HideDoorMover requires -DoorTrial.' }
if ($DoorPassTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $GameFrames -lt 80 -or $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or
    $ObservationGapTrial -or $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial)) {
    throw '-DoorPassTrial requires at least 80 frames, synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($ButtonTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or
    $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial -or $BSPFailureTrial)) {
    throw '-ButtonTrial requires synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($ButtonAutoTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or
    $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $BSPFailureTrial)) {
    throw '-ButtonAutoTrial requires synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($TeammateMemoryTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $GameFrames -lt 70 -or $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or
    $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $ButtonAutoTrial -or $BSPFailureTrial)) {
    throw '-TeammateMemoryTrial requires at least 70 frames, synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($TeammateSearchTrial -and ($Map -ne 'base1' -or $TransitionMap -ne 'base2' -or -not $SynchronizedStart -or
    $GameFrames -lt 70 -or $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or
    $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $ButtonAutoTrial -or
    $TeammateMemoryTrial -or $BSPFailureTrial)) {
    throw '-TeammateSearchTrial requires at least 70 frames, synchronized base1 to base2 transition and no other gameplay trial.'
}
if ($BSPFailureTrial -and ($BSPFailureTrial -notin @('unavailable', 'incomplete') -or
    $Map -ne 'base1' -or $TransitionMap -or -not $SynchronizedStart -or
    $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or
    $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial)) {
    throw '-BSPFailureTrial requires unavailable or incomplete, base1, synchronized start, and no other gameplay trial.'
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
    if ($ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $GroundEdgeTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $ButtonAutoTrial -or $TeammateMemoryTrial -or $TeammateSearchTrial) { $args = "+set cheats 1 $args" }
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
        if ($DoorTrial) {
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '96,-180,24'
        }
        if ($DoorPassTrial) {
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '160,-800,24'
        }
        if ($ButtonTrial) {
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '360,1940,-144'
        }
        if ($ButtonAutoTrial) {
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '194,2080,-144'
        }
        if ($TeammateMemoryTrial) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '220,1940,-144'
            $humanConfig.test.teleport_after = '194,2080,-144'
            $humanConfig.test.teleport_after_frames = 12
        }
        if ($TeammateSearchTrial) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.teleport_map = 'base2'
            $humanConfig.test.teleport = '320,1940,-144'
            $humanConfig.test.teleport_after = '194,2080,-144'
            $humanConfig.test.teleport_after_frames = 3
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
        if ($GroundEdgeTrial) {
            $botConfig.test.teleport_map = 'base1'
            $botConfig.test.teleport = '-88,40,24'
            $botConfig.test.ground_edge_probe = $true
        }
        if ($DoorTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '96,-300,24'
            $botConfig.test.door_probe = $true
            if ($HideDoorMover) { $botConfig.test.hide_door_53 = $true }
        }
        if ($DoorPassTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '-64,-800,24'
            $botConfig.test.door_pass_probe = $true
        }
        if ($ButtonTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '320,1940,-144'
            $botConfig.test.button_probe = $true
        }
        if ($ButtonAutoTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '194,1940,-144'
            $botConfig.test.button_auto_goal = $true
        }
        if ($TeammateMemoryTrial -or $TeammateSearchTrial) {
            $botConfig.test.teleport_map = 'base2'
            $botConfig.test.teleport = '194,1940,-144'
        }
        if ($BSPFailureTrial -eq 'unavailable') { $botConfig.test.no_bsp = $true }
        if ($BSPFailureTrial -eq 'incomplete') { $botConfig.test.partial_bsp = $true }
        if ($NoAASTrial) { $botConfig.test.no_aas = $true }
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
        $groundEdgeBlockedFrames = 0
        $groundEdgeProbeFrames = 0
        $groundEdgeMinZ = [double]::PositiveInfinity
        $groundEdgeMaxOffset = 0.0
        $noAASDirectFrames = 0
        $noAASMoveFrames = 0
        $noAASGroundBlocks = 0
        $noAASOrigin = $null
        $noAASMaxProgress = 0.0
        $doorMoverObserved = $false
        $doorBlockedFrames = 0
        $doorUnobservedFrames = 0
        $doorShotBlockedFrames = 0
        $doorProbeFrames = 0
        $doorMaxY = [double]::NegativeInfinity
        $doorPassBlockedFrames = 0
        $doorPassOpenMoveFrames = 0
        $doorPassMinZ = [double]::PositiveInfinity
        $doorPassMaxZ = [double]::NegativeInfinity
        $doorPassMaxX = [double]::NegativeInfinity
        $doorPassCrossedGrounded = $false
        $bspStatusFrames = 0
        $bspNeutralFrames = 0
        $bspMotionFrames = 0
        $bspOrigin = $null
        $bspMaxDrift = 0.0
        $buttonProbeFrames = 0
        $buttonMoveFrames = 0
        $buttonFireFrames = 0
        $buttonMinY = [double]::PositiveInfinity
        $buttonMaxY = [double]::NegativeInfinity
        $buttonMaxMoverY = [double]::NegativeInfinity
        $buttonMaxDoorZ = [double]::NegativeInfinity
        $buttonFirstMovedFrame = $null
        $buttonDoorFirstMovedFrame = $null
        $autoApproachFrames = 0
        $autoTouchFrames = 0
        $autoFireFrames = 0
        $autoResumedFrames = 0
        $autoMaxButtonY = [double]::NegativeInfinity
        $autoMaxDoorZ = [double]::NegativeInfinity
        $autoButtonFirstFrame = $null
        $autoDoorFirstFrame = $null
        $autoTouched = $false
        $memoryVisibleFrames = 0
        $memoryHiddenFrames = 0
        $memoryHiddenMoveFrames = 0
        $memoryHiddenWaitFrames = 0
        $memoryMaxAge = 0
        $memoryUnexpectedLastPosition = 0
        $searchFrames = 0
        $searchMoveFrames = 0
        $searchStartX = $null
        $searchMaxX = [double]::NegativeInfinity
        $searchWaitAfter = 0
        $searchWaitAtPoint = 0
        $searchWaitMoveFrames = 0
        foreach ($line in Get-Content -LiteralPath $tracePath) {
            $entry = $line | ConvertFrom-Json
            $sent["$($entry.spawncount):$($entry.client_sequence)"] = $entry.sent_command
            if ($null -ne $entry.teammate) { $teammateSeenInTrace = $true }
            if ($TransitionMap -and $entry.map -eq $TransitionMap -and $null -ne $entry.teammate) {
                $teammateSeenAfterTransition = $true
            }
            if ($TeammateMemoryTrial -and $entry.map -eq 'base2') {
                if ($null -ne $entry.teammate -and [math]::Abs($entry.teammate[0] - 220) -lt 32 -and
                    [math]::Abs($entry.teammate[1] - 1940) -lt 32) { $memoryVisibleFrames++ }
                if ($null -eq $entry.teammate -and $null -ne $entry.last_teammate -and $entry.teammate_age_frames -gt 0) {
                    $memoryHiddenFrames++
                    $memoryMaxAge = [math]::Max($memoryMaxAge, [int]$entry.teammate_age_frames)
                    if ([math]::Abs($entry.last_teammate[0] - 220) -gt 32 -or
                        [math]::Abs($entry.last_teammate[1] - 1940) -gt 32) { $memoryUnexpectedLastPosition++ }
                    if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0 -or
                        $entry.sent_command.Up -ne 0) { $memoryHiddenMoveFrames++ }
                    if ($entry.goal -eq 'wait_for_teammate') { $memoryHiddenWaitFrames++ }
                }
            }
            if ($TeammateSearchTrial -and $entry.map -eq 'base2' -and $null -eq $entry.teammate -and
                $null -ne $entry.last_teammate -and [math]::Abs($entry.last_teammate[0] - 320) -lt 32 -and
                [math]::Abs($entry.last_teammate[1] - 1940) -lt 32) {
                if ($entry.goal -eq 'search_last_seen') {
                    $searchFrames++
                    if ($null -eq $searchStartX) { $searchStartX = [double]$entry.self[0] }
                    $searchMaxX = [math]::Max($searchMaxX, [double]$entry.self[0])
                    if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0) { $searchMoveFrames++ }
                }
                if ($entry.teammate_age_frames -gt 40 -and $entry.goal -eq 'wait_for_teammate' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and $entry.sent_command.Up -eq 0) {
                    $searchWaitAfter++
                }
                if ($entry.goal -eq 'wait_for_teammate') {
                    if ($entry.teammate_age_frames -le 40 -and
                        [math]::Sqrt([math]::Pow($entry.self[0] - $entry.last_teammate[0], 2) +
                                     [math]::Pow($entry.self[1] - $entry.last_teammate[1], 2)) -le 64) { $searchWaitAtPoint++ }
                    if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0 -or
                        $entry.sent_command.Up -ne 0) { $searchWaitMoveFrames++ }
                }
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
            if ($GroundEdgeTrial -and $entry.map -eq 'base1' -and $entry.on_ground -and
                [math]::Abs($entry.self[0] + 88) -lt 8 -and [math]::Abs($entry.self[1] - 40) -lt 8) {
                $groundEdgeProbeFrames++
                $groundEdgeMinZ = [math]::Min($groundEdgeMinZ, [double]$entry.self[2])
                $offset = [math]::Sqrt([math]::Pow($entry.self[0] + 88, 2) + [math]::Pow($entry.self[1] - 40, 2))
                $groundEdgeMaxOffset = [math]::Max($groundEdgeMaxOffset, $offset)
                if ($entry.arbitration.limit_reason -eq 'no_ground_support' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and $entry.sent_command.Up -eq 0) {
                    $groundEdgeBlockedFrames++
                }
            }
            if ($NoAASTrial) {
                if ($entry.navigation -eq 'direct_clear') { $noAASDirectFrames++ }
                if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0) { $noAASMoveFrames++ }
                if ($entry.arbitration.limit_reason -eq 'no_ground_support') { $noAASGroundBlocks++ }
                if (-not $GroundEdgeTrial) {
                    if ($null -eq $noAASOrigin) { $noAASOrigin = @([double]$entry.self[0], [double]$entry.self[1]) }
                    $progress = [math]::Sqrt([math]::Pow($entry.self[0] - $noAASOrigin[0], 2) + [math]::Pow($entry.self[1] - $noAASOrigin[1], 2))
                    $noAASMaxProgress = [math]::Max($noAASMaxProgress, $progress)
                }
            }
            if ($DoorTrial -and $entry.map -eq 'base2' -and $entry.on_ground -and
                [math]::Abs($entry.self[0] - 96) -lt 8 -and [math]::Abs($entry.self[1] + 300) -lt 8) {
                $doorProbeFrames++
                $doorMaxY = [math]::Max($doorMaxY, [double]$entry.self[1])
                if (@($entry.movers | Where-Object { $_.model -eq 53 }).Count -gt 0) { $doorMoverObserved = $true }
                if (@($entry.enemies | Where-Object {
                    $_.class -eq 'monster_infantry' -and [math]::Abs($_.origin[0] - 96) -lt 8 -and
                    [math]::Abs($_.origin[1] + 96) -lt 8 -and $_.clear_shot -eq $false
                }).Count -gt 0) { $doorShotBlockedFrames++ }
                if ($entry.arbitration.move_limit_reason -eq 'dynamic_door_blocked' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and $entry.sent_command.Up -eq 0) {
                    $doorBlockedFrames++
                }
                if ($entry.arbitration.move_limit_reason -eq 'dynamic_door_unobserved' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and $entry.sent_command.Up -eq 0) {
                    $doorUnobservedFrames++
                }
            }
            if ($DoorPassTrial -and $entry.map -eq 'base2' -and [math]::Abs($entry.self[1] + 800) -lt 64) {
                $doorPassMaxX = [math]::Max($doorPassMaxX, [double]$entry.self[0])
                if ($entry.self[0] -gt 96 -and $entry.on_ground) { $doorPassCrossedGrounded = $true }
                $doorMover = @($entry.movers | Where-Object { $_.model -eq 27 } | Select-Object -First 1)
                if ($doorMover.Count -gt 0) {
                    $z = [double]$doorMover[0].origin[2]
                    $doorPassMinZ = [math]::Min($doorPassMinZ, $z)
                    $doorPassMaxZ = [math]::Max($doorPassMaxZ, $z)
                    if ($z -gt 72 -and ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0)) { $doorPassOpenMoveFrames++ }
                }
                if ($entry.arbitration.move_limit_reason -eq 'dynamic_door_blocked' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0) { $doorPassBlockedFrames++ }
            }
            if ($BSPFailureTrial -and $entry.map -eq 'base1') {
                if ($entry.geometry_status -eq $BSPFailureTrial) { $bspStatusFrames++ }
                if ($entry.arbitration.limit_reason -eq "bsp_$BSPFailureTrial" -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and
                    $entry.sent_command.Up -eq 0 -and $entry.sent_command.Buttons -eq 0) { $bspNeutralFrames++ }
                if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0 -or
                    $entry.sent_command.Up -ne 0 -or $entry.sent_command.Buttons -ne 0) { $bspMotionFrames++ }
                if ($null -eq $bspOrigin) { $bspOrigin = @([double]$entry.self[0], [double]$entry.self[1]) }
                $drift = [math]::Sqrt([math]::Pow($entry.self[0] - $bspOrigin[0], 2) + [math]::Pow($entry.self[1] - $bspOrigin[1], 2))
                $bspMaxDrift = [math]::Max($bspMaxDrift, $drift)
            }
            if ($ButtonTrial -and $entry.map -eq 'base2' -and $entry.on_ground -and
                [math]::Abs($entry.self[0] - 320) -lt 8 -and $entry.self[1] -ge 1900 -and $entry.self[1] -le 2020) {
                $buttonProbeFrames++
                $buttonMinY = [math]::Min($buttonMinY, [double]$entry.self[1])
                $buttonMaxY = [math]::Max($buttonMaxY, [double]$entry.self[1])
                if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0) { $buttonMoveFrames++ }
                if ($entry.sent_command.Buttons -ne 0) { $buttonFireFrames++ }
                $buttonMover = @($entry.movers | Where-Object { $_.model -eq 34 } | Select-Object -First 1)
                if ($buttonMover.Count -gt 0) {
                    $movedY = [double]$buttonMover[0].origin[1]
                    $buttonMaxMoverY = [math]::Max($buttonMaxMoverY, $movedY)
                    if ($movedY -gt 1 -and $null -eq $buttonFirstMovedFrame) { $buttonFirstMovedFrame = [int]$entry.frame }
                }
                $buttonDoor = @($entry.movers | Where-Object { $_.model -eq 33 } | Select-Object -First 1)
                if ($buttonDoor.Count -gt 0) {
                    $movedZ = [double]$buttonDoor[0].origin[2]
                    $buttonMaxDoorZ = [math]::Max($buttonMaxDoorZ, $movedZ)
                    if ($movedZ -gt 1 -and $null -eq $buttonDoorFirstMovedFrame) { $buttonDoorFirstMovedFrame = [int]$entry.frame }
                }
            }
            if ($ButtonAutoTrial -and $entry.map -eq 'base2' -and $entry.on_ground) {
                if ($entry.arbitration.skill -eq 'button_approach') { $autoApproachFrames++ }
                if ($entry.arbitration.skill -eq 'button_touch') {
                    $autoTouchFrames++
                    $autoTouched = $true
                }
                if ($entry.arbitration.skill -like 'button*' -and $entry.sent_command.Buttons -ne 0) { $autoFireFrames++ }
                if ($autoTouched -and $entry.goal -eq 'follow_teammate' -and
                    ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0)) { $autoResumedFrames++ }
                $buttonMover = @($entry.movers | Where-Object { $_.model -eq 34 } | Select-Object -First 1)
                if ($buttonMover.Count -gt 0) {
                    $movedY = [double]$buttonMover[0].origin[1]
                    $autoMaxButtonY = [math]::Max($autoMaxButtonY, $movedY)
                    if ($movedY -gt 1 -and $null -eq $autoButtonFirstFrame) { $autoButtonFirstFrame = [int]$entry.frame }
                }
                $buttonDoor = @($entry.movers | Where-Object { $_.model -eq 33 } | Select-Object -First 1)
                if ($buttonDoor.Count -gt 0) {
                    $movedZ = [double]$buttonDoor[0].origin[2]
                    $autoMaxDoorZ = [math]::Max($autoMaxDoorZ, $movedZ)
                    if ($movedZ -gt 1 -and $null -eq $autoDoorFirstFrame) { $autoDoorFirstFrame = [int]$entry.frame }
                }
            }
        }
        $humanAtTop = $false
        $memoryHumanBehindWall = $false
        if (($TeammateMemoryTrial -or $TeammateSearchTrial) -and (Test-Path -LiteralPath $humanTracePath)) {
            foreach ($line in Get-Content -LiteralPath $humanTracePath) {
                $entry = $line | ConvertFrom-Json
                if ($entry.map -eq 'base2' -and [math]::Abs($entry.self[0] - 194) -lt 16 -and
                    [math]::Abs($entry.self[1] - 2080) -lt 16) { $memoryHumanBehindWall = $true; break }
            }
        }
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
            geometry_status = $world.geometry_status
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
            ground_edge_trial = [bool]$GroundEdgeTrial; ground_edge_probe_frames = $groundEdgeProbeFrames
            ground_edge_blocked_frames = $groundEdgeBlockedFrames
            ground_edge_min_z = $(if ($GroundEdgeTrial -and $groundEdgeMinZ -ne [double]::PositiveInfinity) { $groundEdgeMinZ } else { $null })
            ground_edge_max_offset = $groundEdgeMaxOffset
            no_aas_trial = [bool]$NoAASTrial; no_aas_direct_frames = $noAASDirectFrames
            no_aas_move_frames = $noAASMoveFrames; no_aas_ground_blocks = $noAASGroundBlocks
            no_aas_max_progress = $noAASMaxProgress
            door_trial = [bool]$DoorTrial; door_mover_observed = $doorMoverObserved
            door_probe_frames = $doorProbeFrames; door_blocked_frames = $doorBlockedFrames
            door_unobserved_trial = [bool]$HideDoorMover; door_unobserved_frames = $doorUnobservedFrames
            door_shot_blocked_frames = $doorShotBlockedFrames
            door_max_y = $(if ($DoorTrial -and $doorMaxY -ne [double]::NegativeInfinity) { $doorMaxY } else { $null })
            door_pass_trial = [bool]$DoorPassTrial; door_pass_blocked_frames = $doorPassBlockedFrames
            door_pass_open_move_frames = $doorPassOpenMoveFrames
            door_pass_min_z = $(if ($DoorPassTrial -and $doorPassMinZ -ne [double]::PositiveInfinity) { $doorPassMinZ } else { $null })
            door_pass_max_z = $(if ($DoorPassTrial -and $doorPassMaxZ -ne [double]::NegativeInfinity) { $doorPassMaxZ } else { $null })
            door_pass_max_x = $(if ($DoorPassTrial -and $doorPassMaxX -ne [double]::NegativeInfinity) { $doorPassMaxX } else { $null })
            door_pass_crossed_grounded = $doorPassCrossedGrounded
            bsp_failure_trial = $BSPFailureTrial; bsp_status_frames = $bspStatusFrames
            bsp_neutral_frames = $bspNeutralFrames; bsp_motion_frames = $bspMotionFrames
            bsp_max_xy_drift = $bspMaxDrift
            button_trial = [bool]$ButtonTrial; button_probe_frames = $buttonProbeFrames
            button_move_frames = $buttonMoveFrames; button_fire_frames = $buttonFireFrames
            button_min_y = $(if ($buttonMinY -ne [double]::PositiveInfinity) { $buttonMinY } else { $null })
            button_max_y = $(if ($buttonMaxY -ne [double]::NegativeInfinity) { $buttonMaxY } else { $null })
            button_max_mover_y = $(if ($buttonMaxMoverY -ne [double]::NegativeInfinity) { $buttonMaxMoverY } else { $null })
            button_max_door_z = $(if ($buttonMaxDoorZ -ne [double]::NegativeInfinity) { $buttonMaxDoorZ } else { $null })
            button_first_moved_frame = $buttonFirstMovedFrame
            button_door_first_moved_frame = $buttonDoorFirstMovedFrame
            button_auto_trial = [bool]$ButtonAutoTrial; button_auto_approach_frames = $autoApproachFrames
            button_auto_touch_frames = $autoTouchFrames; button_auto_fire_frames = $autoFireFrames
            button_auto_resumed_frames = $autoResumedFrames
            button_auto_max_button_y = $(if ($autoMaxButtonY -ne [double]::NegativeInfinity) { $autoMaxButtonY } else { $null })
            button_auto_max_door_z = $(if ($autoMaxDoorZ -ne [double]::NegativeInfinity) { $autoMaxDoorZ } else { $null })
            button_auto_button_first_frame = $autoButtonFirstFrame
            button_auto_door_first_frame = $autoDoorFirstFrame
            teammate_memory_trial = [bool]$TeammateMemoryTrial
            memory_visible_frames = $memoryVisibleFrames; memory_hidden_frames = $memoryHiddenFrames
            memory_max_age_frames = $memoryMaxAge; memory_hidden_move_frames = $memoryHiddenMoveFrames
            memory_hidden_wait_frames = $memoryHiddenWaitFrames; memory_unexpected_last_position = $memoryUnexpectedLastPosition
            memory_human_behind_wall = $memoryHumanBehindWall
            teammate_search_trial = [bool]$TeammateSearchTrial
            search_frames = $searchFrames; search_move_frames = $searchMoveFrames
            search_start_x = $searchStartX
            search_max_x = $(if ($searchMaxX -ne [double]::NegativeInfinity) { $searchMaxX } else { $null })
            search_wait_after = $searchWaitAfter
            search_wait_at_point = $searchWaitAtPoint; search_wait_move_frames = $searchWaitMoveFrames
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
if ($GroundEdgeTrial -and @($results | Where-Object {
    $_.ground_edge_probe_frames -lt 3 -or $_.ground_edge_blocked_frames -lt 3 -or
    $_.ground_edge_blocked_frames -ne $_.ground_edge_probe_frames -or
    $_.ground_edge_min_z -lt 20 -or $_.ground_edge_max_offset -ge 8
}).Count -gt 0) {
    throw "Ground edge trial did not hold at the unsupported step: $summary"
}
if ($NoAASTrial -and @($results | Where-Object {
    $_.aas_loaded -or $_.aas_areas -ne 0 -or $_.aas_reachabilities -ne 0 -or
    $(if ($GroundEdgeTrial) { $_.ground_edge_blocked_frames -lt 3 } else {
        $_.no_aas_direct_frames -lt 3 -or $_.no_aas_move_frames -lt 3 -or $_.no_aas_max_progress -lt 30
    })
}).Count -gt 0) {
    throw "No-AAS trial did not show the expected direct movement or safe edge stop: $summary"
}
if ($DoorTrial -and -not $HideDoorMover -and @($results | Where-Object {
    -not $_.door_mover_observed -or $_.door_probe_frames -lt 3 -or $_.door_blocked_frames -lt 3 -or
    $_.door_shot_blocked_frames -lt 3 -or $_.door_max_y -gt -268
}).Count -gt 0) {
    throw "Door trial did not observe and block the closed door: $summary"
}
if ($HideDoorMover -and @($results | Where-Object {
    $_.door_mover_observed -or $_.door_probe_frames -lt 3 -or $_.door_unobserved_frames -lt 3 -or
    $_.door_shot_blocked_frames -lt 3 -or $_.door_max_y -gt -268
}).Count -gt 0) {
    throw "Unobserved door trial did not hold the unknown passage: $summary"
}
if ($ButtonTrial -and @($results | Where-Object {
    $_.button_probe_frames -lt 10 -or $_.button_move_frames -lt 2 -or $_.button_fire_frames -ne 0 -or
    $_.button_min_y -gt 1941 -or $_.button_max_y -lt 1958 -or
    $_.button_max_mover_y -lt 3 -or $_.button_max_door_z -lt 60 -or
    $null -eq $_.button_first_moved_frame -or $null -eq $_.button_door_first_moved_frame -or
    $_.button_first_moved_frame -ge $_.button_door_first_moved_frame
}).Count -gt 0) {
    throw "Button trial did not touch button *34 and open door *33: $summary"
}
if ($ButtonAutoTrial -and @($results | Where-Object {
    $_.geometry_status -ne 'ready' -or -not $_.aas_loaded -or
    $_.button_auto_approach_frames -lt 3 -or $_.button_auto_touch_frames -lt 3 -or
    $_.button_auto_fire_frames -ne 0 -or $_.button_auto_resumed_frames -lt 1 -or
    $_.button_auto_max_button_y -lt 3 -or $_.button_auto_max_door_z -lt 60 -or
    $null -eq $_.button_auto_button_first_frame -or $null -eq $_.button_auto_door_first_frame -or
    $_.button_auto_button_first_frame -ge $_.button_auto_door_first_frame
}).Count -gt 0) {
    throw "Automatic button choice did not activate door and resume: $summary"
}
if ($TeammateMemoryTrial -and @($results | Where-Object {
    -not $_.memory_human_behind_wall -or $_.memory_visible_frames -lt 3 -or $_.memory_hidden_frames -lt 10 -or $_.memory_max_age_frames -lt 10 -or
    $_.memory_hidden_move_frames -ne 0 -or $_.memory_hidden_wait_frames -lt 10 -or
    $_.memory_unexpected_last_position -ne 0
}).Count -gt 0) {
    throw "Teammate visibility and last-seen memory trial failed: $summary"
}
if ($TeammateSearchTrial -and @($results | Where-Object {
    -not $_.memory_human_behind_wall -or $_.search_frames -lt 2 -or $_.search_move_frames -lt 2 -or
    $null -eq $_.search_start_x -or $null -eq $_.search_max_x -or $_.search_max_x - $_.search_start_x -lt 20 -or
    $_.search_wait_after -lt 5 -or $_.search_wait_at_point -lt 3 -or $_.search_wait_move_frames -ne 0
}).Count -gt 0) {
    throw "Teammate last-seen search trial failed: $summary"
}
if ($DoorPassTrial -and @($results | Where-Object {
    $_.door_pass_blocked_frames -lt 1 -or $_.door_pass_open_move_frames -lt 1 -or
    $_.door_pass_min_z -gt 8 -or $_.door_pass_max_z -lt 72 -or
    $_.door_pass_max_x -lt 96 -or -not $_.door_pass_crossed_grounded
}).Count -gt 0) {
    throw "Door pass trial did not show trigger, opening, and grounded crossing: $summary"
}
if ($BSPFailureTrial -and @($results | Where-Object {
    $_.geometry_status -ne $BSPFailureTrial -or -not $_.aas_loaded -or $_.aas_reachabilities -le 0 -or
    $_.bsp_status_frames -lt $GameFrames -or $_.bsp_neutral_frames -lt $GameFrames -or
    $_.bsp_motion_frames -ne 0 -or $_.bsp_max_xy_drift -ge 2
}).Count -gt 0) {
    throw "BSP failure trial did not hold a neutral command with AAS present: $summary"
}
Write-Output "Saved $summary"
