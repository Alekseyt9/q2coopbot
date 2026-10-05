[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunRoot,[Parameter(Mandatory)][string]$RelayRoot)
$ErrorActionPreference='Stop'
$events=@(Get-Content -LiteralPath (Join-Path $RelayRoot 'feedback.jsonl')|ForEach-Object{ConvertFrom-Json $_})
$steps=@(Get-Content -LiteralPath (Join-Path $RunRoot 'dataset/steps.jsonl')|ForEach-Object{ConvertFrom-Json $_})
$rewards=@(Get-Content -LiteralPath (Join-Path $RunRoot 'dataset/rewards.jsonl')|ForEach-Object{ConvertFrom-Json $_})
$effects=@(Get-Content -LiteralPath (Join-Path $RunRoot 'dataset/server_outcomes.jsonl')|ForEach-Object{ConvertFrom-Json $_})
$initial=Get-Content -LiteralPath (Join-Path $RunRoot 'dataset/episode_start.json') -Raw|ConvertFrom-Json
$mismatches=0
$resetCount=@($events|Where-Object kind -eq reset).Count
if($resetCount -ne 1 -or $events[0].kind -ne 'reset' -or
    ($events[0].initial_observation|ConvertTo-Json -Depth 40 -Compress) -ne ($initial.first_usable_observation|ConvertTo-Json -Depth 40 -Compress) -or
    ($events[0].reset_proof|ConvertTo-Json -Depth 40 -Compress) -ne ($initial.observed_reset_proof|ConvertTo-Json -Depth 40 -Compress)){$mismatches++}
$delivered=@($events|Where-Object kind -eq step)
$expectedIndex=1
foreach($event in $delivered){
    $index=[int]$event.step.index-1
    if($event.step.index -ne $expectedIndex -or $index -lt 0 -or $index -ge $steps.Count -or
        ($event.step|ConvertTo-Json -Depth 40 -Compress) -ne ($steps[$index]|ConvertTo-Json -Depth 40 -Compress) -or
        ($event.reward|ConvertTo-Json -Depth 40 -Compress) -ne ($rewards[$index]|ConvertTo-Json -Depth 40 -Compress) -or
        ($event.effects|ConvertTo-Json -Depth 40 -Compress) -ne ($effects[$index]|ConvertTo-Json -Depth 40 -Compress)){$mismatches++}
    if($expectedIndex -lt $delivered.Count -and ($event.step.terminal -or $event.step.truncated)){$mismatches++}
    $expectedIndex++
}
$last=$delivered|Select-Object -Last 1
if(!$last -or (!$last.step.terminal -and !$last.step.truncated)){$mismatches++}
$report=[pscustomobject]@{version='combat_feedback_v1';accepted=($mismatches -eq 0);resets=$resetCount;steps=$delivered.Count;reward_steps=@($delivered|Where-Object {$_.reward.available}).Count;mismatches=$mismatches;terminal=[bool]$last.step.terminal;truncated=[bool]$last.step.truncated;scope='Exact comparison of delivered first-life prefix with verified offline reset, steps, rewards and native effects.'}
$report|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $RelayRoot 'audit.json') -Encoding utf8NoBOM
$report
if(!$report.accepted){throw 'Delivered feedback differs from verified offline export'}
