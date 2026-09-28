[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$SuiteRoot)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $SuiteRoot).Path
$suite=Get-Content (Join-Path $root 'suite-summary.json') -Raw|ConvertFrom-Json
if($suite.failed -ne 0 -or $suite.passed -ne 2){throw 'Expected two passing native runs'}
$results=@(foreach($run in $suite.results){
    $summary=Get-Content (Join-Path $run.directory 'summary.json') -Raw|ConvertFrom-Json
    $trace=@(Get-Content -LiteralPath $summary.trace_jsonl|ConvertFrom-Json|Where-Object {$_.map -eq 'base3' -and $_.self})
    $entry=@($trace|Where-Object {$_.elevator -in @('board','wait_bottom','ride') -and $_.self[2] -le -400}|Select-Object -First 1)
    $completed=@($trace|Where-Object {$_.elevator -eq 'completed' -and $_.self[2] -ge -240}|Select-Object -First 1)
    $followed=@($trace|Where-Object {$_.episode_frame -gt $completed[0].episode_frame -and $_.goal -eq 'cover_teammate' -and $_.self[2] -ge -240}|Select-Object -First 1)
    if(!$entry.Count -or !$completed.Count -or !$followed.Count -or $entry[0].episode_frame -ge $completed[0].episode_frame -or @($trace|Where-Object {$_.health -lt 100}).Count){throw "Lift-37 behavior failed on port $($run.port)"}
    [pscustomobject]@{port=$run.port;timescale=$run.timescale;entry_frame=$entry[0].episode_frame;completed_frame=$completed[0].episode_frame;follow_frame=$followed[0].episode_frame;trace=$summary.trace_jsonl}
})
@{accepted=$true;cases=$results}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root 'lift37-report.json')
Write-Output "PASS: $root/lift37-report.json"
