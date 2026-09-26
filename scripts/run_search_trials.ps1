[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [string]$AASDir = '',
    [int]$Port = 28460,
    [int[]]$Timescales = @(1, 2),
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
if ($Timescales.Count -eq 0 -or @($Timescales | Where-Object { $_ -lt 1 }).Count -gt 0 -or
    $Port -lt 1024 -or $Port + 3 * $Timescales.Count -gt 65535) {
    throw 'Invalid timescales or port range.'
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ('workspace\artifacts\search-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

# These are reaction fixtures. A scripted return cannot prove that the viewpoint caused discovery.
$cases = @(
    @{ name = 'hidden-sound'; options = @{ HiddenPlayerSoundTrial = $true }; expected = 'not_seen' },
    @{ name = 'late-return'; options = @{ ReacquireTeammate = $true }; expected = 'not_seen' },
    @{ name = 'active-return'; options = @{ ReacquireTeammate = $true; SearchReturnAfterFrames = 7 }; expected = 'reacquired' }
)
$runs = [System.Collections.Generic.List[object]]::new()
for ($i = 0; $i -lt $cases.Count; $i++) {
    $case = $cases[$i]
    $caseRoot = Join-Path $OutputRoot $case.name
    $options = @{
        RuntimeRoot = $RuntimeRoot; AASDir = $AASDir
        TransitionMap = 'base2'; TransitionAfterFrames = 10; GameFrames = 80
        Timescales = $Timescales; SynchronizedStart = $true; UnlimitedLoopbackRate = $true
        TeammateSearchTrial = $true; SearchExpectedOutcome = $case.expected
        Port = $Port + $i * $Timescales.Count; OutputRoot = $caseRoot
    }
    foreach ($key in $case.options.Keys) { $options[$key] = $case.options[$key] }
    & (Join-Path $PSScriptRoot 'run_speed_trial.ps1') @options
    $summaryPath = Join-Path $caseRoot 'summary.json'
    foreach ($result in @(Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json)) {
        $runs.Add([pscustomobject]@{
            scenario = $case.name; map = $result.final_map; timescale = $result.timescale
            expected_outcome = $case.expected
            attempts = $result.search_attempts; details = $result.search_attempt_details
            scripted_return = $result.search_return_scripted
            discovery_caused_by_search = $null
            matched_commands = $result.matched_applied_commands
            wall_seconds = $result.wall_seconds; game_fps = $result.game_fps
            frame_gaps = $result.frame_gaps; decode_errors = $result.decode_errors
            summary_json = $summaryPath; trace_jsonl = $result.trace_jsonl
        })
    }
}
$report = [ordered]@{
    scope = 'scripted_base2_regression'
    multi_map_validated = $false
    runs = @($runs.ToArray())
}
$reportPath = Join-Path $OutputRoot 'search-report.json'
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$runs | Select-Object scenario, timescale, attempts, matched_commands, frame_gaps, decode_errors | Format-Table -AutoSize
Write-Host "Saved $reportPath"
