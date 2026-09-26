[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [string]$ServerExe = '',
    [string]$ClientExe = '',
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
    [switch]$ReacquireTeammate,
    [int]$SearchReturnAfterFrames = 15,
    [ValidateSet('not_seen', 'reacquired', 'no_new_visibility')][string]$SearchExpectedOutcome = 'not_seen',
    [switch]$HiddenPlayerSoundTrial,
    [string]$SearchFixture = '',
    [string]$ActorScenario = '',
    [int]$ScenarioTailFrames = 0,
    [switch]$SearchWaitBaseline,
    [switch]$SearchApproachOnly,
    [switch]$SearchSynchronizedSetup,
    [string]$BSPFailureTrial = '',
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($ScenarioTailFrames -ne 0 -and (-not $ActorScenario -or $ScenarioTailFrames -lt 2 -or $ScenarioTailFrames -gt 1000)) { throw 'Scenario tail requires an actor scenario and 2..1000 frames.' }
$fixture = $null
$actorScenarioDefinition = $null
if ($ActorScenario) {
    $ActorScenario = [IO.Path]::GetFullPath($ActorScenario)
    $actorScenarioDefinition = Get-Content -LiteralPath $ActorScenario -Raw | ConvertFrom-Json
    if (-not $SynchronizedStart -or $SearchFixture -or $TeammateSearchTrial -or $TeammateMemoryTrial -or $ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $ObservationGapTrial -or $GroundEdgeTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $ButtonAutoTrial -or $NoAASTrial -or $BSPFailureTrial) { throw 'Actor scenario requires synchronized start and no other gameplay trial.' }
    if ($actorScenarioDefinition.map -ne $(if ($TransitionMap) {$TransitionMap} else {$Map}) -or $GameFrames -lt $actorScenarioDefinition.game_frames) {throw 'Scenario map/frame budget mismatch.'}
}
if ($SearchWaitBaseline -and -not $SearchFixture) { throw '-SearchWaitBaseline requires -SearchFixture.' }
if ($SearchSynchronizedSetup -and -not $TeammateSearchTrial) { throw 'Search synchronized setup requires -TeammateSearchTrial.' }
if ($SearchApproachOnly -and -not $SearchFixture -and (-not $TeammateSearchTrial -or -not $HiddenPlayerSoundTrial)) { throw '-SearchApproachOnly requires -SearchFixture or the hidden-sound search trial.' }
if ($SearchApproachOnly -and $SearchWaitBaseline) { throw 'Choose either approach-only or waiting baseline.' }
if ($SearchFixture) {
    $fixture = Get-Content -LiteralPath $SearchFixture -Raw | ConvertFrom-Json
    $fixtureMap = if ($TransitionMap) { $TransitionMap } else { $Map }
    if (-not $SynchronizedStart -or $fixture.map -ne $fixtureMap -or $GameFrames -lt 100 -or
        $ElevatorTrial -or $CombatMoveTrial -or $ObservationGapTrial -or $FriendlyFireTrial -or
        $GroundEdgeTrial -or $NoAASTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or
        $ButtonAutoTrial -or $TeammateMemoryTrial -or $TeammateSearchTrial -or $BSPFailureTrial) {
        throw '-SearchFixture requires a matching final map, synchronized start, at least 100 frames and no other gameplay trial.'
    }
    foreach ($field in @('name', 'bot_origin', 'human_origin', 'human_walk_target')) {
        if (-not $fixture.$field) { throw "Search fixture is missing $field." }
    }
    if ($fixture.walk_after_frames -lt 25 -or $fixture.walk_frames -lt 1 -or
        $fixture.walk_after_frames + $fixture.walk_frames + 10 -gt $GameFrames) {
        throw 'Search fixture walking must finish at least 10 frames before the episode ends.'
    }
}
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
if ($ReacquireTeammate -and -not $TeammateSearchTrial) { throw '-ReacquireTeammate requires -TeammateSearchTrial.' }
if ($SearchReturnAfterFrames -ne 15 -and -not $ReacquireTeammate) { throw '-SearchReturnAfterFrames requires -ReacquireTeammate.' }
if ($SearchReturnAfterFrames -lt 1) { throw '-SearchReturnAfterFrames must be positive.' }
if ($SearchExpectedOutcome -eq 'reacquired' -and -not $ReacquireTeammate) { throw '-SearchExpectedOutcome reacquired requires -ReacquireTeammate.' }
if ($HiddenPlayerSoundTrial -and (-not $TeammateSearchTrial -or $ReacquireTeammate)) { throw '-HiddenPlayerSoundTrial requires -TeammateSearchTrial without -ReacquireTeammate.' }
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
if ($ClientExe) {
    $botExe = (Resolve-Path -LiteralPath $ClientExe).Path
} else {
Push-Location $repoRoot
try {
    & go build -o $botExe ./cmd/q2coopbot
    if ($LASTEXITCODE -ne 0) { throw 'Go build failed.' }
} finally { Pop-Location }
}

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
    $scenarioResultPath = Join-Path $OutputRoot "$name-scenario-completion.json"
    if ($ScenarioTailFrames -and (Test-Path -LiteralPath $scenarioResultPath)) {throw 'Scenario completion already exists; use a fresh output directory.'}
    $stdout = Join-Path $OutputRoot "$name-server.log"
    $stderr = Join-Path $OutputRoot "$name-server.err.log"
    $humanLog = Join-Path $OutputRoot "$name-human.log"
    $humanErr = Join-Path $OutputRoot "$name-human.err.log"
    $humanTracePath = Join-Path $OutputRoot "$name-human-trace.jsonl"
    $botLog = Join-Path $OutputRoot "$name-bot.log"
    $tracePath = Join-Path $OutputRoot "$name-trace.jsonl"
    $worldPath = Join-Path $OutputRoot "$name-world.json"
    $appliedPath = Join-Path $OutputRoot "$name-applied.jsonl"
    # Bind the test server to loopback; Yamagi's default "localhost" binds all interfaces.
    $args = "+set ip 127.0.0.1 +set noipx 1 +set game baseq2 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 4 +set port $runPort +set timescale $scale +map $Map"
    $rconPassword = if ($TransitionMap) { [guid]::NewGuid().ToString('N') } else { '' }
    if ($TransitionMap) { $args = "+set rcon_password $rconPassword $args" }
    if ($UnlimitedLoopbackRate) { $args = "+set sv_test_unlimited_loopback 1 $args" }
    if ($ElevatorTrial -or $CombatMoveTrial -or $FriendlyFireTrial -or $GroundEdgeTrial -or $DoorTrial -or $DoorPassTrial -or $ButtonTrial -or $ButtonAutoTrial -or $TeammateMemoryTrial -or $TeammateSearchTrial -or $SearchFixture -or $ActorScenario) { $args = "+set cheats 1 $args" }
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
        $serverEndpoints = @(Get-NetUDPEndpoint -OwningProcess $server.Id -ErrorAction SilentlyContinue)
        if (-not ($serverEndpoints | Where-Object { $_.LocalPort -eq $runPort -and $_.LocalAddress -eq '127.0.0.1' })) {
            throw 'Test server did not bind its expected loopback UDP endpoint.'
        }
        if ($serverEndpoints | Where-Object { $_.LocalAddress -notin @('127.0.0.1', '::1') }) {
            throw 'Test server opened a non-loopback UDP endpoint.'
        }
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
            $humanConfig.test.teleport_after = $(if ($HiddenPlayerSoundTrial) { '191,2111,-103' } else { '194,2080,-144' })
            $humanConfig.test.teleport_after_frames = 3
            if ($HiddenPlayerSoundTrial) {
                $humanConfig.test.jump_after_teleport_frames = 25
                $humanConfig.test.jump_again_after_teleport_frames = 33
            }
            if ($ReacquireTeammate) {
                $humanConfig.test.teleport_return = '400,1840,-144'
                $humanConfig.test.teleport_return_after_frames = $SearchReturnAfterFrames
            }
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
        if ($actorScenarioDefinition) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.scenario = $ActorScenario
            if ($ScenarioTailFrames) { $humanConfig.test.scenario_result = $scenarioResultPath }
            $humanConfig.test.teleport_map = $actorScenarioDefinition.map
            $humanConfig.test.teleport = ($actorScenarioDefinition.actor_origin | ForEach-Object {([double]$_).ToString([cultureinfo]::InvariantCulture)}) -join ','
        }
        if ($SearchSynchronizedSetup) { $humanConfig.test.scenario_frame_origin = 40 }
        if ($fixture) {
            $humanConfig.output.trace_jsonl = $humanTracePath
            $humanConfig.test.teleport_map = $fixture.map
            $humanConfig.test.teleport = $fixture.human_origin
            $humanConfig.test.walk_target = $fixture.human_walk_target
			$humanConfig.test.walk_route = [bool]$fixture.walk_route
            $humanConfig.test.walk_after_frames = [int]$fixture.walk_after_frames
            $humanConfig.test.walk_frames = [int]$fixture.walk_frames
            $humanConfig.test.scenario_frame_origin = [int]$fixture.frame_origin
        }
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
        if ($SearchApproachOnly) { $botConfig.test.disable_probe = $true }
        if ($actorScenarioDefinition) {
            $botConfig.test.teleport_map = $actorScenarioDefinition.map
			if ($actorScenarioDefinition.bot_invulnerable) { $botConfig.test.invulnerable=$true }
			if ($actorScenarioDefinition.bot_health) { $botConfig.test.initial_health=[int]$actorScenarioDefinition.bot_health }
			if ($actorScenarioDefinition.map_entry) { $botConfig.test.change_entry=$actorScenarioDefinition.map_entry }
            $botConfig.test.teleport = ($actorScenarioDefinition.bot_origin | ForEach-Object {([double]$_).ToString([cultureinfo]::InvariantCulture)}) -join ','
            $botConfig.test.scenario_frame_origin = $actorScenarioDefinition.start_frame
            $botConfig.test.setup_hold_frames = 1
            if ($ScenarioTailFrames) { $botConfig.test.scenario_result=$scenarioResultPath; $botConfig.test.scenario_tail_frames=$ScenarioTailFrames }
        }
        if ($SearchSynchronizedSetup) {
            $botConfig.test.scenario_frame_origin = 40
            $botConfig.test.setup_hold_frames = 4
        }
        if ($fixture) {
            $botConfig.test.teleport_map = $fixture.map
            $botConfig.test.teleport = $fixture.bot_origin
            $botConfig.test.disable_search = [bool]$SearchWaitBaseline
            $botConfig.test.setup_hold_frames = [int]$fixture.bot_hold_frames
            $botConfig.test.scenario_frame_origin = [int]$fixture.frame_origin
        }
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
        $earlyStop = $false
        $stopEvidence = Get-Content -LiteralPath $botLog | Select-String 'scenario_stop_frame=(\d+) scenario_terminal_frame=(\d+)' | Select-Object -Last 1
        if ($stopEvidence) {
            if (-not $ScenarioTailFrames) {throw 'Unexpected early scenario stop.'}
            $completion = Get-Content -LiteralPath $scenarioResultPath -Raw | ConvertFrom-Json
            $match=$stopEvidence.Matches[0]
            if ([int]$match.Groups[2].Value -ne $completion.end_frame -or [int]$match.Groups[1].Value -lt $completion.end_frame+$ScenarioTailFrames) {throw 'Scenario stop did not cover required tail.'}
            $lastBotRow=Get-Content -LiteralPath $tracePath -Tail 1 | ConvertFrom-Json
            if ($lastBotRow.frame -ne [int]$match.Groups[1].Value -or $lastBotRow.map -ne $completion.map -or $lastBotRow.spawncount -ne $completion.generation) {throw 'Scenario completion does not match final bot trace.'}
            $earlyStop=$true
        }
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
        $searchFireFrames = 0
        $searchStartX = $null
        $searchMaxX = [double]::NegativeInfinity
        $searchWaitAfter = 0
        $searchWaitAtPoint = 0
        $searchWaitMoveFrames = 0
        $probeFrames = 0
        $probeMoveFrames = 0
        $probeFireFrames = 0
        $probeTarget = $null
        $probeTargetChanges = 0
        $searchReacquiredFrames = 0
        $reacquiredFollowMoveFrames = 0
        $probeAfterReacquire = 0
        $probeCompletedFrames = 0
        $searchAttemptStarts = @{}
        $searchAttemptOutcomes = @{}
        $searchAttemptDetails = @{}
        $searchAttemptLastSelf = @{}
        $searchAttemptInvalid = 0
        $soundEvents = 0
        $soundExplicitPositions = 0
        $soundEntityOnly = 0
        $hiddenPlayerJumpSounds = 0
        $hiddenPlayerJumpPositions = 0
        $hiddenPlayerJumpEntity = $null
        $hiddenPlayerJumpFrame = $null
        $soundCueFrames = 0
        $soundCueZeroAge = 0
        $soundCueMaxAge = -1
        $soundCueWrongEntity = 0
        $soundCueInvalid = 0
        $soundCueExpiredFrames = 0
        $soundPHSPossibleClusters = $null
        $soundPHSTotalClusters = $null
        $evidenceActivityMax = 0
        $evidenceActivityFramesMax = 0
        $evidenceReacquireEvents = 0
        $evidenceInvalid = 0
        $evidenceVisualOutside = 0
        $motionFrames = 0
        $motionInvalid = 0
        $motionExpiredFrames = 0
        $motionAtSoundNearby = $null
        $motionAtSoundTotal = $null
        $motionAtSoundRadius = $null
        foreach ($line in Get-Content -LiteralPath $tracePath) {
            $entry = $line | ConvertFrom-Json
            $sent["$($entry.spawncount):$($entry.client_sequence)"] = $entry.sent_command
            foreach ($sound in @($entry.sounds)) {
                if ($null -eq $sound) { continue }
                $soundEvents++
                if ($null -ne $sound.position) { $soundExplicitPositions++ }
                elseif ($sound.entity -gt 0) { $soundEntityOnly++ }
                if ($HiddenPlayerSoundTrial -and $entry.map -eq 'base2' -and $null -eq $entry.teammate -and
                    $sound.name -match '(^|/)\*?jump1\.wav$' -and $sound.entity -gt 0) {
                    $hiddenPlayerJumpSounds++
                    $hiddenPlayerJumpEntity = [int]$sound.entity
                    $hiddenPlayerJumpFrame = [int]$entry.frame
                    if ($null -ne $sound.position) { $hiddenPlayerJumpPositions++ }
                }
            }
            if ($HiddenPlayerSoundTrial -and $entry.map -eq 'base2') {
                if ($null -ne $entry.teammate_motion) {
                    $motionFrames++
                    $expectedRadius = 32 + 40 * [int]$entry.teammate_motion.age_frames
                    if ($null -ne $entry.teammate -or
                        $entry.teammate_motion.age_frames -ne $entry.teammate_age_frames -or
                        $entry.teammate_motion.radius -ne $expectedRadius -or
                        $entry.teammate_motion.nearby_ground_areas -gt $entry.teammate_motion.total_ground_areas -or
                        $entry.teammate_motion.method -ne 'conditional_horizontal_radius') {
                        $motionInvalid++
                    }
                } elseif ($null -eq $entry.teammate -and $entry.teammate_age_frames -gt 40) {
                    $motionExpiredFrames++
                }
                if ($null -ne $entry.teammate_sound) {
                    $soundCueFrames++
                    $soundCueMaxAge = [math]::Max($soundCueMaxAge, [int]$entry.teammate_sound.age_frames)
                    if ($entry.teammate_sound.age_frames -eq 0) { $soundCueZeroAge++ }
                    if ($entry.teammate_sound.age_frames -eq 0 -and $null -ne $entry.teammate_sound.phs_total_clusters) {
                        $soundPHSPossibleClusters = [int]$entry.teammate_sound.phs_possible_clusters
                        $soundPHSTotalClusters = [int]$entry.teammate_sound.phs_total_clusters
                    }
                    if ($entry.teammate_sound.age_frames -eq 0 -and $null -ne $entry.teammate_motion) {
                        $motionAtSoundNearby = [int]$entry.teammate_motion.nearby_ground_areas
                        $motionAtSoundTotal = [int]$entry.teammate_motion.total_ground_areas
                        $motionAtSoundRadius = [double]$entry.teammate_motion.radius
                    }
                    if ($null -ne $hiddenPlayerJumpEntity -and $entry.teammate_sound.entity -ne $hiddenPlayerJumpEntity) {
                        $soundCueWrongEntity++
                    }
                    if ($null -ne $entry.teammate -or $entry.teammate_sound.age_frames -lt 0 -or
                        $entry.teammate_sound.age_frames -gt 10 -or $null -ne $entry.teammate_sound.position -or
                        $null -ne $entry.teammate_sound.source_areas) {
                        $soundCueInvalid++
                    }
                } elseif ($null -ne $hiddenPlayerJumpFrame -and $entry.frame -gt $hiddenPlayerJumpFrame + 10 -and
                    $null -eq $entry.teammate) {
                    $soundCueExpiredFrames++
                }
            }
            if ($null -ne $entry.teammate) { $teammateSeenInTrace = $true }
            if (($TeammateSearchTrial -or $SearchFixture) -and $null -ne $entry.search_attempt) {
                $attempt = $entry.search_attempt
                $attemptKey = "$($entry.spawncount):$($entry.map):$($attempt.entity):$($attempt.last_seen_frame):$($attempt.start_frame)"
                $searchAttemptStarts[$attemptKey] = $true
                if ($attempt.state -eq 'completed') { $searchAttemptOutcomes[$attemptKey] = $attempt.outcome }
                if (-not $searchAttemptDetails.ContainsKey($attemptKey)) {
                    $searchAttemptDetails[$attemptKey] = [pscustomobject]@{
                        map = $entry.map; spawncount = $entry.spawncount
                        start_frame = $attempt.start_frame; end_frame = $null
                        duration_frames = 0; travel_horizontal_units = 0.0
                        state = 'active'; outcome = $null; visible_at_end = $null
                        visibility = $attempt.visibility
                    }
                }
                $detail = $searchAttemptDetails[$attemptKey]
                if ($detail.state -eq 'active') {
                    if ($searchAttemptLastSelf.ContainsKey($attemptKey)) {
                        $previousSelf = $searchAttemptLastSelf[$attemptKey]
                        $detail.travel_horizontal_units += [math]::Sqrt(
                            [math]::Pow($entry.self[0] - $previousSelf[0], 2) +
                            [math]::Pow($entry.self[1] - $previousSelf[1], 2))
                    }
                    $searchAttemptLastSelf[$attemptKey] = $entry.self
                    $detail.duration_frames = [int]$entry.frame - [int]$attempt.start_frame
                    if ($attempt.state -eq 'completed') {
                        $detail.state = 'completed'; $detail.outcome = $attempt.outcome
                        $detail.end_frame = $attempt.end_frame
                        $detail.visible_at_end = $null -ne $entry.teammate
                        if ($attempt.end_frame -ne $entry.frame -or
                            ($attempt.outcome -eq 'reacquired' -and -not $detail.visible_at_end)) {
                            $searchAttemptInvalid++
                        }
                    }
                }
                if ($attempt.attempt -ne 1 -or $attempt.max_attempts -ne 1 -or
                    $attempt.basis -ne 'last_seen_aas_viewpoint' -or
                    $attempt.expected_observation -ne 'teammate_visible_in_current_snapshot') {
                    $searchAttemptInvalid++
                }
            }
            if ($null -ne $entry.teammate_evidence) {
                if ($null -eq $entry.teammate) {
                    $evidenceActivityMax = [math]::Max($evidenceActivityMax, [int]$entry.teammate_evidence.activity_sounds)
                    $evidenceActivityFramesMax = [math]::Max($evidenceActivityFramesMax, [int]$entry.teammate_evidence.activity_frames)
                    if ($entry.teammate_evidence.location_status -ne 'unknown' -or
                        $entry.teammate_evidence.reacquired -or $entry.teammate_evidence.visual_outside_nominal) {
                        $evidenceInvalid++
                    }
                } elseif ($entry.teammate_evidence.reacquired) {
                    $evidenceReacquireEvents++
                    if ($entry.teammate_evidence.location_status -ne 'observed') { $evidenceInvalid++ }
                    if ($entry.teammate_evidence.visual_outside_nominal) { $evidenceVisualOutside++ }
                }
            }
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
                    if ($entry.sent_command.Buttons -ne 0) { $searchFireFrames++ }
                }
                if ($entry.goal -eq 'probe_last_seen') {
                    $probeFrames++
                    if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0) { $probeMoveFrames++ }
                    if ($entry.sent_command.Buttons -ne 0) { $probeFireFrames++ }
                    $target = @([double]$entry.search_target[0], [double]$entry.search_target[1], [double]$entry.search_target[2])
                    if ($null -eq $probeTarget) { $probeTarget = $target }
                    elseif ([math]::Abs($target[0] - $probeTarget[0]) -gt 1 -or
                            [math]::Abs($target[1] - $probeTarget[1]) -gt 1) { $probeTargetChanges++ }
                }
                if ($entry.teammate_age_frames -gt 40 -and $entry.goal -eq 'wait_for_teammate' -and
                    $entry.sent_command.Forward -eq 0 -and $entry.sent_command.Side -eq 0 -and $entry.sent_command.Up -eq 0) {
                    $searchWaitAfter++
                }
                if ($entry.goal -eq 'wait_for_teammate') {
                    if ($probeFrames -gt 0 -and $null -ne $probeTarget -and $entry.teammate_age_frames -le 40 -and
                        [math]::Sqrt([math]::Pow($entry.self[0] - $probeTarget[0], 2) +
                                     [math]::Pow($entry.self[1] - $probeTarget[1], 2)) -le 16) { $probeCompletedFrames++ }
                    if ($entry.teammate_age_frames -le 40 -and
                        [math]::Sqrt([math]::Pow($entry.self[0] - $entry.last_teammate[0], 2) +
                                     [math]::Pow($entry.self[1] - $entry.last_teammate[1], 2)) -le 64) { $searchWaitAtPoint++ }
                    if ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0 -or
                        $entry.sent_command.Up -ne 0) { $searchWaitMoveFrames++ }
                }
            }
            if ($TeammateSearchTrial -and $entry.map -eq 'base2' -and $searchFrames -gt 0 -and
                $null -ne $entry.teammate -and $entry.teammate_age_frames -eq 0) {
                $searchReacquiredFrames++
                if ($entry.goal -eq 'follow_teammate' -and
                    ($entry.sent_command.Forward -ne 0 -or $entry.sent_command.Side -ne 0)) { $reacquiredFollowMoveFrames++ }
            }
            if ($searchReacquiredFrames -gt 0 -and $entry.goal -eq 'probe_last_seen') { $probeAfterReacquire++ }
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
        $memoryHumanReturned = $false
        $humanEntity = $null
        $humanJumpCommands = 0
        $humanJumpMaxZ = [double]::NegativeInfinity
        if (($TeammateMemoryTrial -or $TeammateSearchTrial) -and (Test-Path -LiteralPath $humanTracePath)) {
            foreach ($line in Get-Content -LiteralPath $humanTracePath) {
                $entry = $line | ConvertFrom-Json
                if ($HiddenPlayerSoundTrial -and $entry.map -eq 'base2') {
                    $humanEntity = [int]$entry.self_entity
                    if ($entry.sent_command.Up -gt 0) { $humanJumpCommands++ }
                    if ([math]::Abs($entry.self[0] - 191) -lt 16 -and [math]::Abs($entry.self[1] - 2111) -lt 16) {
                        $humanJumpMaxZ = [math]::Max($humanJumpMaxZ, [double]$entry.self[2])
                    }
                }
                $hiddenX = $(if ($HiddenPlayerSoundTrial) { 191 } else { 194 })
                $hiddenY = $(if ($HiddenPlayerSoundTrial) { 2111 } else { 2080 })
                if ($entry.map -eq 'base2' -and [math]::Abs($entry.self[0] - $hiddenX) -lt 16 -and
                    [math]::Abs($entry.self[1] - $hiddenY) -lt 16) { $memoryHumanBehindWall = $true }
                if ($entry.map -eq 'base2' -and [math]::Abs($entry.self[0] - 400) -lt 16 -and
                    [math]::Abs($entry.self[1] - 1840) -lt 16) { $memoryHumanReturned = $true }
            }
        }
        $motionOutsideFrames = 0
        if ($HiddenPlayerSoundTrial -and (Test-Path -LiteralPath $humanTracePath)) {
            $humanByFrame = @{}
            foreach ($line in Get-Content -LiteralPath $humanTracePath) {
                $humanEntry = $line | ConvertFrom-Json
                if ($humanEntry.map -eq 'base2') { $humanByFrame[[int]$humanEntry.frame] = $humanEntry }
            }
            foreach ($line in Get-Content -LiteralPath $tracePath) {
                $botEntry = $line | ConvertFrom-Json
                if ($null -eq $botEntry.teammate_motion -or -not $humanByFrame.ContainsKey([int]$botEntry.frame)) { continue }
                $humanEntry = $humanByFrame[[int]$botEntry.frame]
                $dx = [double]$humanEntry.self[0] - [double]$botEntry.last_teammate[0]
                $dy = [double]$humanEntry.self[1] - [double]$botEntry.last_teammate[1]
                if ([math]::Sqrt($dx * $dx + $dy * $dy) -gt [double]$botEntry.teammate_motion.radius) {
                    $motionOutsideFrames++
                }
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
            scenario_early_stop = $earlyStop
            scenario_tail_frames = $ScenarioTailFrames
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
            search_frames = $searchFrames; search_move_frames = $searchMoveFrames; search_fire_frames = $searchFireFrames
            search_start_x = $searchStartX
            search_max_x = $(if ($searchMaxX -ne [double]::NegativeInfinity) { $searchMaxX } else { $null })
            search_wait_after = $searchWaitAfter
            search_wait_at_point = $searchWaitAtPoint; search_wait_move_frames = $searchWaitMoveFrames
            probe_frames = $probeFrames; probe_move_frames = $probeMoveFrames; probe_fire_frames = $probeFireFrames
            probe_target = $probeTarget; probe_target_changes = $probeTargetChanges
            search_attempts = $searchAttemptStarts.Count
            search_attempt_not_seen = @($searchAttemptOutcomes.Values | Where-Object { $_ -eq 'not_seen' }).Count
            search_attempt_reacquired = @($searchAttemptOutcomes.Values | Where-Object { $_ -eq 'reacquired' }).Count
            search_attempt_invalid = $searchAttemptInvalid
            search_attempt_details = @($searchAttemptDetails.Values | Sort-Object start_frame)
            search_fixture_name = $(if ($fixture) { $fixture.name } else { $null })
            search_disabled = [bool]$SearchWaitBaseline
            search_probe_disabled = [bool]$SearchApproachOnly
            search_synchronized_setup = [bool]$SearchSynchronizedSetup
            human_trace_jsonl = $(if (Test-Path -LiteralPath $humanTracePath) { $humanTracePath } else { $null })
            search_return_scripted = [bool]$ReacquireTeammate
            search_return_after_frames = $(if ($ReacquireTeammate) { $SearchReturnAfterFrames } else { $null })
            search_expected_outcome = $SearchExpectedOutcome
            search_reacquired_frames = $searchReacquiredFrames
            probe_completed_frames = $probeCompletedFrames
            sound_events = $soundEvents; sound_explicit_positions = $soundExplicitPositions; sound_entity_only = $soundEntityOnly
            hidden_player_sound_trial = [bool]$HiddenPlayerSoundTrial
            hidden_player_jump_sounds = $hiddenPlayerJumpSounds; hidden_player_jump_positions = $hiddenPlayerJumpPositions
            hidden_player_jump_entity = $hiddenPlayerJumpEntity; hidden_player_jump_frame = $hiddenPlayerJumpFrame
            sound_cue_frames = $soundCueFrames; sound_cue_zero_age = $soundCueZeroAge
            sound_cue_max_age = $soundCueMaxAge; sound_cue_wrong_entity = $soundCueWrongEntity
            sound_cue_invalid = $soundCueInvalid; sound_cue_expired_frames = $soundCueExpiredFrames
            sound_phs_possible_clusters = $soundPHSPossibleClusters; sound_phs_total_clusters = $soundPHSTotalClusters
            evidence_activity_max = $evidenceActivityMax; evidence_activity_frames_max = $evidenceActivityFramesMax
            evidence_reacquire_events = $evidenceReacquireEvents
            evidence_invalid = $evidenceInvalid; evidence_visual_outside = $evidenceVisualOutside
            motion_frames = $motionFrames; motion_invalid = $motionInvalid; motion_expired_frames = $motionExpiredFrames
            motion_at_sound_nearby = $motionAtSoundNearby; motion_at_sound_total = $motionAtSoundTotal
            motion_at_sound_radius = $motionAtSoundRadius
            motion_hidden_teleport_outside_frames = $motionOutsideFrames
            human_entity = $humanEntity; human_jump_commands = $humanJumpCommands
            human_jump_max_z = $(if ($humanJumpMaxZ -ne [double]::NegativeInfinity) { $humanJumpMaxZ } else { $null })
            reacquire_teammate = [bool]$ReacquireTeammate
            memory_human_returned = $memoryHumanReturned
            reacquired_follow_move_frames = $reacquiredFollowMoveFrames; probe_after_reacquire = $probeAfterReacquire
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
if (@($results | Where-Object { ($_.game_frames -lt $GameFrames -and -not $_.scenario_early_stop) -or $_.frame_gaps -gt 0 -or $_.decode_errors -gt 0 -or -not $_.teammate_seen }).Count -gt 0) {
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
    -not $_.memory_human_behind_wall -or $_.search_frames -lt 1 -or $_.search_move_frames -lt 1 -or $_.search_fire_frames -ne 0 -or
    $null -eq $_.search_start_x -or $null -eq $_.search_max_x -or
    $(if ($SearchApproachOnly) {
        $_.probe_frames -ne 0 -or $_.search_attempts -ne 0 -or $null -ne $_.probe_target -or $_.search_wait_at_point -lt 1
    } elseif ($SearchExpectedOutcome -eq 'no_new_visibility') {
        $_.probe_frames -ne 0 -or $_.search_attempts -ne 1 -or $_.search_attempt_invalid -ne 0 -or
        @($_.search_attempt_details | Where-Object { $_.outcome -eq 'no_new_visibility' -and $_.duration_frames -eq 0 -and $_.travel_horizontal_units -eq 0 -and
            $_.visibility.hidden_samples -gt 0 -and $_.visibility.safe_candidates -gt 0 -and $_.visibility.max_newly_visible -eq 0 -and
            $_.visibility.audit_complete -eq $true -and $_.visibility.audit_samples -gt 0 -and $_.visibility.audit_max_gain -eq 0 }).Count -ne 1
    } else {
        $_.probe_frames -lt 2 -or $_.probe_move_frames -lt 2 -or $_.probe_fire_frames -ne 0 -or
        $_.search_attempts -ne 1 -or $_.search_attempt_invalid -ne 0 -or
        $(if ($SearchExpectedOutcome -eq 'reacquired') {
            $_.search_attempt_reacquired -ne 1 -or $_.search_attempt_not_seen -ne 0
        } else {
            $_.search_attempt_not_seen -ne 1 -or $_.search_attempt_reacquired -ne 0 -or $_.probe_completed_frames -lt 1
        }) -or
        $null -eq $_.probe_target -or $_.probe_target_changes -ne 0
    }) -or
    $(if ($ReacquireTeammate) {
        -not $_.memory_human_returned -or $_.search_reacquired_frames -lt 3 -or
        $_.reacquired_follow_move_frames -lt 1 -or $_.probe_after_reacquire -ne 0 -or
        $_.evidence_reacquire_events -lt 1 -or $_.evidence_invalid -ne 0
    } else { $_.search_wait_after -lt 5 -or $_.search_reacquired_frames -ne 0 }) -or
    $_.search_wait_move_frames -ne 0
}).Count -gt 0) {
    throw "Teammate last-seen search trial failed: $summary"
}
if ($HiddenPlayerSoundTrial -and @($results | Where-Object {
    -not $_.memory_human_behind_wall -or $_.human_jump_commands -lt 1 -or $_.human_jump_max_z -lt -90 -or
    $_.hidden_player_jump_sounds -lt 2 -or $_.hidden_player_jump_entity -ne $_.human_entity -or
    $_.hidden_player_jump_positions -ne 0 -or $_.sound_cue_frames -lt 1 -or
    $_.sound_cue_zero_age -lt 2 -or $_.sound_cue_max_age -gt 10 -or
    $_.sound_cue_wrong_entity -ne 0 -or $_.sound_cue_invalid -ne 0 -or
    $_.sound_cue_expired_frames -lt 1 -or
    $_.sound_phs_possible_clusters -lt 1 -or
    $_.sound_phs_total_clusters -le $_.sound_phs_possible_clusters -or
    $_.motion_frames -lt 1 -or $_.motion_invalid -ne 0 -or $_.motion_expired_frames -lt 1 -or
    $_.motion_at_sound_nearby -lt 1 -or $_.motion_at_sound_total -lt $_.motion_at_sound_nearby -or
    $_.motion_hidden_teleport_outside_frames -lt 1 -or
    $_.evidence_activity_max -lt 2 -or $_.evidence_activity_frames_max -lt 2 -or
    $_.evidence_invalid -ne 0
}).Count -gt 0) {
    throw "Hidden player sound trial did not confirm a bounded activity cue: $summary"
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
