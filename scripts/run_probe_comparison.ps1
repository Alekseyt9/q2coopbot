[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [int]$Port = 28650,
    [int[]]$Timescales = @(1, 2),
    [switch]$ReportOnly,
    [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
if ($ReportOnly -and -not $OutputRoot) { throw '-ReportOnly requires an existing -OutputRoot.' }
if ($ReportOnly) {
    $previous = Get-Content -LiteralPath (Join-Path $OutputRoot 'probe-comparison.json') -Raw | ConvertFrom-Json
    $Timescales = @($previous.pairs | ForEach-Object { [int]$_.timescale })
}
if ($Timescales.Count -eq 0 -or @($Timescales | Select-Object -Unique).Count -ne $Timescales.Count -or
    @($Timescales | Where-Object { $_ -lt 1 }).Count -gt 0 -or
    $Port -lt 1024 -or $Port + 2 * $Timescales.Count -gt 65535) { throw 'Invalid timescales or ports.' }
if (-not $OutputRoot) { $OutputRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ('workspace\artifacts\probe-comparison-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

function Distance($a,$b) {
    return [math]::Sqrt([math]::Pow($a[0]-$b[0],2)+[math]::Pow($a[1]-$b[1],2)+[math]::Pow($a[2]-$b[2],2))
}
function Read-Window($path) {
    $entries = @(Get-Content -LiteralPath $path | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.map -eq 'base2' -and $_.frame -ge 42 -and $_.frame -lt 102 })
    if ($entries.Count -ne 60) { throw "Incomplete comparison window: $path" }
    for ($i=0; $i -lt 60; $i++) { if ($entries[$i].frame -ne 42+$i) { throw "Frame gap/duplicate: $path" } }
    return $entries
}
$runs = [System.Collections.Generic.List[object]]::new()
$windows = @{}
foreach ($mode in @('probe','approach')) {
    $caseRoot = Join-Path $OutputRoot $mode
    if (-not $ReportOnly) {
        $options = @{
            RuntimeRoot=$RuntimeRoot; Port=$Port+$runs.Count; Timescales=$Timescales
            TransitionMap='base2'; TransitionAfterFrames=10; GameFrames=100
            SynchronizedStart=$true; UnlimitedLoopbackRate=$true; TeammateSearchTrial=$true
            HiddenPlayerSoundTrial=$true; SearchSynchronizedSetup=$true
            SearchApproachOnly=($mode -eq 'approach'); OutputRoot=$caseRoot
        }
        & (Join-Path $PSScriptRoot 'run_speed_trial.ps1') @options
    }
    $summaries = @(Get-Content -LiteralPath (Join-Path $caseRoot 'summary.json') -Raw | ConvertFrom-Json)
    if ($summaries.Count -ne $Timescales.Count -or
        @($summaries.timescale | Select-Object -Unique).Count -ne $Timescales.Count -or
        @($summaries | Where-Object { $_.timescale -notin $Timescales }).Count) { throw 'Unexpected trial set.' }
    foreach ($summary in $summaries) {
        if (-not $summary.search_synchronized_setup -or [bool]$summary.search_probe_disabled -ne ($mode -eq 'approach') -or
            $summary.frame_gaps -ne 0 -or $summary.decode_errors -ne 0 -or $summary.search_attempt_invalid -ne 0) {
            throw 'Wrong scenario mode or invalid trace.'
        }
        $bot = @(Read-Window $summary.trace_jsonl)
        $human = @(Read-Window $summary.human_trace_jsonl)
        $windows["$mode-$($summary.timescale)"] = @{ bot=$bot; human=$human }
        $distance=0.0
        for ($i=1; $i -lt $bot.Count; $i++) {
            $distance += [math]::Sqrt([math]::Pow($bot[$i].self[0]-$bot[$i-1].self[0],2)+[math]::Pow($bot[$i].self[1]-$bot[$i-1].self[1],2))
        }
        $probe = @($bot | Where-Object { $_.goal -eq 'probe_last_seen' })
        $loss = @($bot | Where-Object { $null -eq $_.teammate -and $null -ne $_.last_teammate } | Select-Object -First 1)
        if ($loss.Count -ne 1) { throw 'Player did not become hidden.' }
        $rediscovered = @($bot | Where-Object { $_.frame -gt $loss[0].frame -and $null -ne $_.teammate }).Count -gt 0
        if ($mode -eq 'approach' -and ($probe.Count -ne 0 -or $summary.search_attempts -ne 0)) { throw 'Approach baseline performed a probe.' }
        $runs.Add([pscustomobject]@{
            mode=$mode; timescale=$summary.timescale; loss_frame=$loss[0].frame
            first_probe_frame=$(if ($probe.Count) { $probe[0].frame } else { $null })
            approach_frames=@($bot | Where-Object { $_.goal -eq 'search_last_seen' }).Count
            probe_frames=$probe.Count; attempts=$summary.search_attempts; attempt_details=$summary.search_attempt_details
            visibility=$(if ($probe.Count) { $probe[0].search_attempt.visibility } else { $null })
            reacquired=$rediscovered; bot_travel_horizontal=$distance
            bot_min_health=($bot | Measure-Object -Property health -Minimum).Minimum
            hidden_player_jump_sounds=$summary.hidden_player_jump_sounds
            matched_commands=$summary.matched_applied_commands
            trace_jsonl=$summary.trace_jsonl; human_trace_jsonl=$summary.human_trace_jsonl
        })
    }
}
$pairs = @($Timescales | ForEach-Object {
    $scale=$_
    $probe=@($runs | Where-Object { $_.mode -eq 'probe' -and $_.timescale -eq $scale })[0]
    $approach=@($runs | Where-Object { $_.mode -eq 'approach' -and $_.timescale -eq $scale })[0]
    $a=$windows["probe-$scale"]; $b=$windows["approach-$scale"]
    $humanDelta=0.0; $prefixDelta=0.0; $prefixCommandsEqual=$true
    for ($i=0; $i -lt 60; $i++) {
        $humanDelta=[math]::Max($humanDelta,(Distance $a.human[$i].self $b.human[$i].self))
        if ($a.bot[$i].frame -le $probe.first_probe_frame) {
            $prefixDelta=[math]::Max($prefixDelta,(Distance $a.bot[$i].self $b.bot[$i].self))
            if ($a.bot[$i].frame -lt $probe.first_probe_frame -and
                (($a.bot[$i].sent_command | ConvertTo-Json -Compress) -ne ($b.bot[$i].sent_command | ConvertTo-Json -Compress))) { $prefixCommandsEqual=$false }
        }
    }
    $comparable=$null -ne $probe.first_probe_frame -and $humanDelta -le 0.25 -and $prefixDelta -le 0.25 -and
        $prefixCommandsEqual -and $probe.loss_frame -eq $approach.loss_frame
    [pscustomobject]@{
        timescale=$scale; comparable=$comparable; human_path_max_delta=$humanDelta
        pre_probe_bot_max_delta=$prefixDelta; pre_probe_commands_equal=$prefixCommandsEqual
        extra_travel_horizontal=$probe.bot_travel_horizontal-$approach.bot_travel_horizontal
        probe_reacquired=$probe.reacquired; approach_reacquired=$approach.reacquired
        conclusion=$(if (-not $comparable) {'inconclusive_fixture_mismatch'}
            elseif (-not $probe.reacquired -and -not $approach.reacquired) {'probe_added_travel_without_reacquisition'}
            elseif ($probe.reacquired -and -not $approach.reacquired) {'probe_only_reacquisition_in_fixture'}
            else {'compare_reacquisition_timing'})
    }
})
$report=[ordered]@{ scope='base2_hidden_sound_probe_ablation'; initial_and_hidden_placement='scripted_teleport'; combat_state_controlled=$false; evaluation_start_frame=42; evaluation_frames=60; runs=@($runs.ToArray()); pairs=$pairs }
$path=Join-Path $OutputRoot 'probe-comparison.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $path -Encoding UTF8
$pairs | Format-Table -AutoSize
Write-Host "Saved $path"
if (@($pairs | Where-Object { -not $_.comparable }).Count) { throw 'Probe comparison conditions differ; see probe-comparison.json.' }
