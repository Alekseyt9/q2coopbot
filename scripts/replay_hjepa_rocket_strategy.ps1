[CmdletBinding(DefaultParameterSetName = 'Replay')]
param(
    [Parameter(ParameterSetName = 'Replay')]
    [Parameter(ParameterSetName = 'Compare')]
    [ValidateRange(1, 2147483647)]
    [int] $Seed = 1835166139,

    [Parameter(ParameterSetName = 'Replay')]
    [ValidateSet('retry', 'cancel', 'original-route', 'relocate', 'fire')]
    [string] $Strategy = 'retry',

    [Parameter(Mandatory, ParameterSetName = 'Compare')]
    [switch] $Compare,

    [Parameter(Mandatory, ParameterSetName = 'List')]
    [switch] $List
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archive = Join-Path $repo 'workspace/artifacts/hjepa-rocket-movement-window-20261126'
$trial = Join-Path $archive 'rocket-release-retry'
$runtime = Join-Path $trial 'runtime'
$catalogPath = Join-Path $trial 'confirmation/cohort.json'
$protocolPath = Join-Path $trial 'confirmation/native-protocol.json'
$manifestPath = Join-Path $runtime 'manifest.json'
$riskPath = Join-Path $archive 'rocket-closed-loop-release/runtime-weights.json'
$fitPath = Join-Path $archive 'rocket-release-splash/weights.json'
$fitProtocolPath = Join-Path $archive 'rocket-release-splash/protocol.json'

function Read-ReplayJson([string] $Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
}

function Confirm-ReplayHash([string] $Path, [string] $Expected) {
    if ($Expected -notmatch '^[a-fA-F0-9]{64}$') {
        throw "Missing SHA-256 for '$Path'."
    }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected) {
        throw "SHA-256 mismatch for '$Path'."
    }
}

if (!(Test-Path -LiteralPath $protocolPath -PathType Leaf)) {
    throw "Local H-JEPA replay artifacts are missing at '$trial'."
}
$protocol = Read-ReplayJson $protocolPath
Confirm-ReplayHash $catalogPath $protocol.source
$catalog = Read-ReplayJson $catalogPath

if ($List) {
    $catalog.cases.source | Sort-Object map, seed | Select-Object map, seed, profile
    return
}

$selected = @($catalog.cases | Where-Object { $_.source.seed -eq $Seed })
if ($selected.Count -ne 1) {
    throw "Seed $Seed is absent from the verified catalog. Use -List to view available seeds."
}
$source = $selected[0].source
$sourceRoot = $source.source
Confirm-ReplayHash (Join-Path $sourceRoot 'combat-checkpoint/manifest.json') $source.manifest_sha256
Confirm-ReplayHash (Join-Path $sourceRoot 'bot.jsonl') $selected[0].trace_sha256
Confirm-ReplayHash $manifestPath $protocol.runtime
Confirm-ReplayHash $riskPath $protocol.weights
$runtimeManifest = Read-ReplayJson $manifestPath
foreach ($file in 'q2coopbot.exe', 'q2checkpoint.exe', 'q2hjepa-fragment-proof.exe', 'run-branch.ps1') {
    Confirm-ReplayHash (Join-Path $runtime $file) $runtimeManifest.binaries[$file]
}

$risk = Read-ReplayJson $riskPath
Confirm-ReplayHash $fitPath $risk.source_sha256
$fit = Read-ReplayJson $fitPath
Confirm-ReplayHash $fitProtocolPath $fit.protocol_sha256
$fitProtocol = Read-ReplayJson $fitProtocolPath
$sourceConfig = Read-ReplayJson (Join-Path $sourceRoot 'bot-config.json')
Confirm-ReplayHash $sourceConfig.combat.provider_file $fitProtocol.actor_sha256

$streams = @{
    retry = [int]$protocol.streams.retry_replan
    cancel = [int]$protocol.streams.cancel
    'original-route' = [int]$protocol.streams.retry
    relocate = 158
    fire = 156
}
$replays = Join-Path $repo 'workspace/artifacts/hjepa-rocket-strategy-replays'

function Invoke-Replay([string] $Mode, [string] $Run) {
    $stream = $streams[$Mode]
    New-Item -ItemType Directory -Path $run | Out-Null
    $branch = Join-Path $run 'branch'

    $socket = [Net.Sockets.UdpClient]::new([Net.IPEndPoint]::new([Net.IPAddress]::Loopback, 0))
    try {
        $port = $socket.Client.LocalEndPoint.Port
    } finally {
        $socket.Dispose()
    }

    $environment = @{
        Q2COOPBOT_IMPACT_SHIFT_ASSETS = (Join-Path $sourceRoot 'runtime/baseq2')
        Q2COOPBOT_EXPERIMENTAL_ROCKET_RISK = $riskPath
        Q2COOP_COMBAT_OPTION_PAK_POOL = (Join-Path $repo 'workspace/build')
        GOMAXPROCS = '2'
    }
    $previous = @{}
    Push-Location -LiteralPath $repo
    try {
        foreach ($key in $environment.Keys) {
            $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
            [Environment]::SetEnvironmentVariable($key, $environment[$key], 'Process')
        }
        $nativeLog = Join-Path $run 'native.log'
        $branchArgs = @(
            '-NoProfile', '-File', (Join-Path $runtime 'run-branch.ps1'),
            '-SourceRoot', $sourceRoot, '-OutputRoot', $branch,
            '-Client', (Join-Path $runtime 'q2coopbot.exe'),
            '-CheckpointTool', (Join-Path $runtime 'q2checkpoint.exe'),
            '-Port', $port, '-Frames', 120, '-SamplingStream', $stream,
            '-SamplingScope', 'root_committed_plan4_v1'
        )
        & pwsh @branchArgs *> $nativeLog
        if ($LASTEXITCODE -ne 0) {
            throw "Local replay failed. See '$nativeLog'."
        }

        $prove = Join-Path $runtime 'q2hjepa-fragment-proof.exe'
        $proofArgs = @('--source', $sourceRoot, '--branch', $branch, '--repo', $repo)
        $rootPath = Join-Path $run 'root.json'
        & $prove @proofArgs --root-only 1> $rootPath 2> (Join-Path $run 'root.err')
        if ($LASTEXITCODE -ne 0) {
            throw "Root verification failed in '$run'."
        }
        $root = Read-ReplayJson $rootPath
        if (!$root.accepted -or $root.sampling_stream -ne $stream -or
                $root.model -ne ('hjepa:' + $fitProtocol.actor_sha256)) {
            throw 'The restored root did not match the selected H-JEPA strategy.'
        }

        $tailPath = Join-Path $run 'tail.json'
        $receiptPath = Join-Path $run 'tail-receipt.json'
        & $prove @proofArgs --frames 120 --export-fragment $tailPath 1> $receiptPath 2> (Join-Path $run 'tail.err')
        if ($LASTEXITCODE -ne 0) {
            throw "Trajectory verification failed in '$run'."
        }
        $receipt = Read-ReplayJson $receiptPath
        if (!$receipt.accepted) {
            throw "The native trajectory was not accepted in '$run'."
        }
        Confirm-ReplayHash $tailPath $receipt.data_sha256
    } finally {
        foreach ($key in $previous.Keys) {
            [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
        }
        Pop-Location
    }

    $tail = Read-ReplayJson $tailPath
    $life = $tail.native_root
    $end = $tail.transitions[-1].next_identity.frame
    $rocketFrames = @(
        foreach ($line in Get-Content -LiteralPath (Join-Path $branch 'server.log')) {
            if ($line -match '^sv_test_projectile spawncount=(\d+) server_frame=(\d+) .*event=spawn .*attacker=(\d+) mod=8(?: |$)') {
                $frame = [int]$Matches[2]
                if ([long]$Matches[1] -eq $life.spawncount -and [int]$Matches[3] -eq $life.actor -and
                        $frame -ge $life.frame -and $frame -le $end) {
                    $frame
                }
            }
        }
    )
    $strategyEvents = @(
        foreach ($line in Get-Content -LiteralPath (Join-Path $branch 'bot.err')) {
            if ($line -match 'combat_option ((?:first_attempt|first_attempt_release|release_risk|release_retry|release_retry_replan|rank_blaster|persistent_equipped) frame=(\d+)(?: .*)?)$' -and
                    [int]$Matches[2] -ge $life.frame -and [int]$Matches[2] -lt $end) {
                $Matches[1]
            }
        }
    )
    $selfDamage = 0
    $monsterDamage = 0
    foreach ($step in $tail.transitions) {
        foreach ($hit in $step.native_effects) {
            if ($hit.attacker -eq $life.actor -and $hit.target -eq $life.actor) {
                $selfDamage += [math]::Max(0, [math]::Min($hit.health_before, $hit.take))
            } elseif ($hit.attacker -eq $life.actor -and $hit.target_class -like 'monster_*') {
                $monsterDamage += [math]::Max(0, [math]::Min($hit.health_before, $hit.take))
            }
        }
    }
    $summary = [ordered]@{
        map = $life.map
        seed = $Seed
        strategy = $Mode
        native_first_life_transitions = $receipt.rows
        native_boundary = $receipt.boundary
        first_life_trace_self_hp = $selfDamage
        first_life_trace_monster_hp = $monsterDamage
        last_observed_health = $tail.transitions[-1].next_health
        rocket_spawn_frames = $rocketFrames
        strategy_events = $strategyEvents
        output = $run
    }
    $summaryJson = $summary | ConvertTo-Json -Depth 6
    $summaryJson | Set-Content -LiteralPath (Join-Path $run 'summary.json') -Encoding utf8NoBOM
    $summary
}

function Assert-TracePrefix($Expected, $Actual, [int] $Count) {
    if ($Expected.Count -lt $Count -or $Actual.Count -lt $Count) {
        throw 'An alternative ended before the common decision.'
    }
    $fields = 'features', 'next_features', 'sent_command', 'logical_sampling_identity', 'next_health', 'terminal'
    for ($i = 0; $i -lt $Count; $i++) {
        foreach ($field in $fields) {
            $a = ConvertTo-Json -InputObject $Expected[$i][$field] -Depth 32 -Compress
            $b = ConvertTo-Json -InputObject $Actual[$i][$field] -Depth 32 -Compress
            if ($a -cne $b) {
                throw "Replay transition $i differs in field '$field'."
            }
        }
    }
}

if (!$Compare) {
    $artifactName = '{0}-{1}-{2}' -f $Seed, $Strategy, [guid]::NewGuid().ToString('N')
    Invoke-Replay $Strategy (Join-Path $replays $artifactName) | ConvertTo-Json -Depth 6
    return
}

$comparisonRoot = Join-Path $replays ('{0}-comparison-{1}' -f $Seed, [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $comparisonRoot | Out-Null
$modes = 'cancel', 'original-route', 'retry'
$summaries = [ordered]@{}
$traces = [ordered]@{}
$firstRisk = [ordered]@{}
foreach ($mode in $modes) {
    $summary = Invoke-Replay $mode (Join-Path $comparisonRoot $mode)
    $summaries[$mode] = $summary
    $traces[$mode] = (Read-ReplayJson (Join-Path $summary.output 'tail.json')).transitions
    $firstRisk[$mode] = $summary.strategy_events | Where-Object { $_ -like 'release_risk *' } | Select-Object -First 1
}

$reference = $traces.cancel
$first = $firstRisk.cancel
$frame = $null
$decision = 'no_attempt'
$shared = $reference.Count
if ($first) {
    if ($first -notmatch '^release_risk frame=(\d+) .* allow=(true|false)$') {
        throw 'The comparison did not log a recognized risk decision.'
    }
    $frame = [int]$Matches[1]
    $decision = if ($Matches[2] -eq 'true') { 'allowed' } else { 'veto' }
    if ($decision -eq 'veto') {
        $shared = @($reference | Where-Object { $_.identity.frame -le $frame }).Count
        if ($shared -eq 0 -or $reference[$shared - 1].identity.frame -ne $frame) {
            throw 'The first veto does not correspond to a verified transition.'
        }
    }
}
foreach ($mode in 'original-route', 'retry') {
    if ($firstRisk[$mode] -cne $first) {
        throw "The first risk decision differs in '$mode'."
    }
    if ($decision -ne 'veto' -and $traces[$mode].Count -ne $reference.Count) {
        throw "Replay length differs in '$mode' without a veto."
    }
    Assert-TracePrefix $reference $traces[$mode] $shared
}

$comparison = [ordered]@{
    seed = $Seed
    map = $source.map
    first_risk_decision = $decision
    first_risk_frame = $frame
    shared_transitions_verified = $shared
    strategies = $summaries
    output = $comparisonRoot
}
$json = $comparison | ConvertTo-Json -Depth 9
$json | Set-Content -LiteralPath (Join-Path $comparisonRoot 'comparison.json') -Encoding utf8NoBOM
$json
