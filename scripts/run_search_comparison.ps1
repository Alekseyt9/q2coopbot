[CmdletBinding()]
param(
    [string]$Fixture = (Join-Path $PSScriptRoot 'scenarios\search-base1-walk.json'),
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [int]$Port = 28520,
    [int[]]$Timescales = @(1, 2),
    [switch]$ReportOnly,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
if ($ReportOnly -and -not $OutputRoot) { throw '-ReportOnly requires an existing -OutputRoot.' }
if ($ReportOnly) {
    $previousReport = Get-Content -LiteralPath (Join-Path $OutputRoot 'comparison.json') -Raw | ConvertFrom-Json
    $scenario = $previousReport.fixture
    $Timescales = @($previousReport.pairs | ForEach-Object { [int]$_.timescale })
} else {
    $scenario = Get-Content -LiteralPath $Fixture -Raw | ConvertFrom-Json
}
if ($Timescales.Count -eq 0 -or @($Timescales | Where-Object { $_ -lt 1 }).Count -gt 0 -or
    $Port -lt 1024 -or $Port + 2 * $Timescales.Count -gt 65535) { throw 'Invalid timescales or ports.' }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ('workspace\artifacts\search-comparison-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$evaluationFrames = if ($scenario.evaluation_frames) { [int]$scenario.evaluation_frames } else { 40 }
if ($evaluationFrames -lt 20) { throw 'Comparison requires at least 20 evaluation frames.' }
if (-not $ReportOnly) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $Fixture = Join-Path $OutputRoot 'fixture.json'
    $scenario | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Fixture -Encoding UTF8
}

function Distance($a, $b) {
    return [math]::Sqrt([math]::Pow($a[0] - $b[0], 2) + [math]::Pow($a[1] - $b[1], 2) + [math]::Pow($a[2] - $b[2], 2))
}

function Measure-Run($summary, $mode) {
    $humanConfig = Get-Content -LiteralPath $summary.human_config_json -Raw | ConvertFrom-Json
    $botConfig = Get-Content -LiteralPath $summary.bot_config_json -Raw | ConvertFrom-Json
    if ($humanConfig.test.teleport -ne $scenario.human_origin -or $humanConfig.test.walk_target -ne $scenario.human_walk_target -or
        $humanConfig.test.walk_after_frames -ne $scenario.walk_after_frames -or $humanConfig.test.walk_frames -ne $scenario.walk_frames -or
        $humanConfig.test.scenario_frame_origin -ne $scenario.frame_origin -or $botConfig.test.scenario_frame_origin -ne $scenario.frame_origin -or
        $botConfig.test.teleport -ne $scenario.bot_origin -or $botConfig.test.setup_hold_frames -ne $scenario.bot_hold_frames -or
        [bool]$botConfig.test.disable_search -ne ($mode -eq 'wait')) {
        throw 'Recorded client configs do not match the comparison fixture.'
    }
    if (-not $summary.aas_loaded -or $summary.geometry_status -ne 'ready') { throw 'Search comparison requires AAS and complete BSP geometry.' }
    $bot = @(Get-Content -LiteralPath $summary.trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq $scenario.map })
    $human = @(Get-Content -LiteralPath $summary.human_trace_jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.map -eq $scenario.map })
    $walk = @($human | Where-Object { $_.arbitration.move_source -eq 'test_walk' })
    $held = @($bot | Where-Object { $_.arbitration.limit_reason -eq 'test_setup_hold' })
    if ($held.Count -eq 0 -or $walk.Count -ne $scenario.walk_frames) { throw 'Missing setup or incomplete walking trace.' }
    $evaluationStart = [int]$held[-1].frame + 1
    $evaluation = @($bot | Where-Object { $_.frame -ge $evaluationStart -and $_.frame -lt $evaluationStart + $evaluationFrames })
    $hidden = @($evaluation | Where-Object { $null -eq $_.teammate -and $null -ne $_.last_teammate })
    $target = @($scenario.human_walk_target.Split(',') | ForEach-Object { [double]::Parse($_, [cultureinfo]::InvariantCulture) })
    if ($evaluation.Count -ne $evaluationFrames -or $hidden.Count -eq 0 -or
        @($walk | Where-Object { -not $_.on_ground -or $_.health -le 0 }).Count -gt 0 -or
        (Distance $walk[-1].self $target) -gt 16) {
        throw 'Fixture did not produce grounded walking to the target and a hidden player during evaluation.'
    }
    for ($i = 1; $i -lt $walk.Count; $i++) {
        if ($walk[$i].frame -ne $walk[$i-1].frame + 1 -or (Distance $walk[$i].self $walk[$i-1].self) -gt 40) {
            throw 'Human walking trace has a frame gap or discontinuous movement.'
        }
    }
    $firstLoss = @($bot | Where-Object { $_.frame -ge $walk[0].frame -and $null -eq $_.teammate -and $null -ne $_.last_teammate })[0]
    if ($null -eq $firstLoss -or @($bot | Where-Object { $_.frame -ge $scenario.frame_origin -and $_.frame -lt $walk[0].frame -and $null -ne $_.teammate }).Count -lt 5) {
        throw 'Fixture lacks confirmed visibility before walking and subsequent loss.'
    }
    $reacquired = @($evaluation | Where-Object { $_.frame -gt $firstLoss.frame -and $null -ne $_.teammate } | Select-Object -First 1)
    $travel = 0.0
    for ($i = 1; $i -lt $evaluation.Count; $i++) {
        $travel += [math]::Sqrt([math]::Pow($evaluation[$i].self[0] - $evaluation[$i-1].self[0], 2) +
            [math]::Pow($evaluation[$i].self[1] - $evaluation[$i-1].self[1], 2))
    }
    $hiddenMoves = @($hidden | Where-Object { $_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0 -or $_.sent_command.Up -ne 0 }).Count
    if ($mode -eq 'wait' -and ($summary.search_attempts -ne 0 -or $hiddenMoves -ne 0)) { throw 'Waiting baseline issued search or movement.' }
    if ($summary.search_attempt_invalid -ne 0) { throw 'Invalid search-attempt telemetry.' }
    return [pscustomobject]@{
        mode = $mode; timescale = $summary.timescale; map = $scenario.map
        evaluation_frames = $evaluationFrames
        evaluation_start_frame = $evaluation[0].frame; start_position = $evaluation[0].self
        loss_age_at_start = $evaluation[0].frame - $firstLoss.frame
        human_walk_start_frame = $walk[0].frame
        human_position_at_release = @($human | Where-Object { $_.frame -eq $evaluation[0].frame } | Select-Object -First 1)[0].self
        hidden_frames = $hidden.Count; hidden_move_frames = $hiddenMoves
        approach_frames = @($evaluation | Where-Object { $_.goal -eq 'search_last_seen' }).Count
        unreachable_approach_frames = @($evaluation | Where-Object { $_.goal -eq 'search_last_seen' -and $_.navigation -eq 'unreachable' }).Count
        probe_attempts = $summary.search_attempts; attempt_details = $summary.search_attempt_details
        reacquired = $reacquired.Count -gt 0
        reacquire_after_loss_frames = $(if ($reacquired.Count) { $reacquired[0].frame - $firstLoss.frame } else { $null })
        bot_travel_horizontal = $travel
        bot_initial_health = $evaluation[0].health
        bot_min_health = ($evaluation | Measure-Object -Property health -Minimum).Minimum
        human_target_error = Distance $walk[-1].self $target
        matched_commands = $summary.matched_applied_commands; frame_gaps = $summary.frame_gaps; decode_errors = $summary.decode_errors
        trace_jsonl = $summary.trace_jsonl; human_trace_jsonl = $summary.human_trace_jsonl
        human_walk_positions = @($walk | ForEach-Object { ,$_.self })
    }
}

$runs = [System.Collections.Generic.List[object]]::new()
foreach ($mode in @('search', 'wait')) {
    $options = @{
        RuntimeRoot = $RuntimeRoot; Map = $scenario.map; GameFrames = 100
        Timescales = $Timescales; Port = $Port + $runs.Count
        SynchronizedStart = $true; UnlimitedLoopbackRate = $true
        SearchFixture = $Fixture; SearchWaitBaseline = ($mode -eq 'wait')
        OutputRoot = (Join-Path $OutputRoot $mode)
    }
    if (-not $ReportOnly) { & (Join-Path $PSScriptRoot 'run_speed_trial.ps1') @options }
    foreach ($summary in @(Get-Content -LiteralPath (Join-Path $options.OutputRoot 'summary.json') -Raw | ConvertFrom-Json)) {
        $runs.Add((Measure-Run $summary $mode))
    }
}
$pairs = @($Timescales | ForEach-Object {
    $scale = $_
    $search = @($runs | Where-Object { $_.mode -eq 'search' -and $_.timescale -eq $scale })[0]
    $wait = @($runs | Where-Object { $_.mode -eq 'wait' -and $_.timescale -eq $scale })[0]
    $maxDelta = 0.0
    for ($i = 0; $i -lt $search.human_walk_positions.Count; $i++) {
        $maxDelta = [math]::Max($maxDelta, (Distance $search.human_walk_positions[$i] $wait.human_walk_positions[$i]))
    }
    $comparable = $maxDelta -le 0.25 -and (Distance $search.start_position $wait.start_position) -le 0.25 -and
        $null -ne $search.human_position_at_release -and $null -ne $wait.human_position_at_release -and
        (Distance $search.human_position_at_release $wait.human_position_at_release) -le 0.25 -and
        $search.loss_age_at_start -eq $wait.loss_age_at_start
    [pscustomobject]@{
        timescale = $scale; comparable = $comparable; human_path_max_delta = $maxDelta
        search_reacquired = $search.reacquired; wait_reacquired = $wait.reacquired
        conclusion = $(if (-not $comparable) { 'inconclusive_fixture_mismatch' }
            elseif (-not $search.reacquired -and -not $wait.reacquired) { 'no_reacquisition_in_either_arm' }
            elseif ($search.reacquired -and -not $wait.reacquired) { 'search_only_reacquisition_in_fixture' }
            else { 'compare_reacquisition_timing' })
    }
})
$report = [ordered]@{ fixture = $scenario; scope = 'scripted_walk_comparison'; runs = @($runs.ToArray()); pairs = $pairs }
$reportPath = Join-Path $OutputRoot 'comparison.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$pairs | Format-Table -AutoSize
Write-Host "Saved $reportPath"
if (@($pairs | Where-Object { -not $_.comparable }).Count -gt 0) { throw 'Paired conditions differ; comparison is inconclusive. See comparison.json.' }
