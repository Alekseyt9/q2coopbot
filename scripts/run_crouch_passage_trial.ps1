[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29360,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if (!$OutputRoot) {$OutputRoot=Join-Path $repo ('workspace/artifacts/crouch-passage-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
& "$PSScriptRoot/run_speed_trial.ps1" -ActorScenario (Join-Path $PSScriptRoot 'scenarios/base1-low-ceiling-crouch.json') -Map base1 -GameFrames 180 -SynchronizedStart -UnlimitedLoopbackRate -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach ($scale in $Timescales) {
    $file=@(Get-ChildItem $OutputRoot -Filter "scale-$scale-port-*-trace.jsonl" | Where-Object Name -NotLike '*human*')
    if ($file.Count -ne 1) {throw 'Expected one bot trace per scale'}
    $rows=@(Get-Content $file[0].FullName | ConvertFrom-Json)
    $request=@($rows | Where-Object {$_.arbitration.skill -eq 'crouch_passage' -and $_.sent_command.Up -lt 0 -and [math]::Abs($_.self[0]+462) -lt 2 -and [math]::Abs($_.self[1]+35) -lt 2 -and [math]::Abs($_.self[2]+23.875) -lt 1} | Select-Object -First 1)
    $duck=@($rows | Where-Object {$request.Count -and $_.frame -gt $request[0].frame -and $_.ducked -and $_.on_ground} | Select-Object -First 1)
    $exit=@($rows | Where-Object {$duck.Count -and $_.frame -gt $duck[0].frame -and !$_.ducked -and $_.on_ground -and $_.self[0] -gt -422 -and [math]::Abs($_.self[2]+23.875) -lt 2} | Select-Object -First 1)
    $bad=@($rows | Where-Object {$request.Count -and $exit.Count -and $_.frame -ge $request[0].frame -and $_.frame -le $exit[0].frame -and ($_.health -le 0 -or $_.map -ne 'base1' -or $_.sent_command.Up -gt 0)})
    [pscustomobject]@{timescale=$scale;accepted=($request.Count -eq 1 -and $duck.Count -eq 1 -and $exit.Count -eq 1 -and $bad.Count -eq 0);scope='prepared_start_under_ceiling_then_exit';request_frame=$request[0].frame;duck_frame=$duck[0].frame;exit_frame=$exit[0].frame;trace=$file[0].FullName}
})
$report | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputRoot 'report.json')
$report
if (@($report | Where-Object {!$_.accepted}).Count) {throw 'Crouch passage not confirmed'}
