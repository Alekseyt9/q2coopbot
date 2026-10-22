param([Parameter(Mandatory)][string]$Evaluation,[Parameter(Mandatory)][string]$Out)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/combat_goal_stop.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
$pool=Get-Content -LiteralPath (Join-Path $Evaluation 'pool/report.json') -Raw|ConvertFrom-Json
if($pool.state -ne 'complete'){throw 'Closed native pool required'}
$checks=@()
foreach($job in $pool.jobs){
    $capture=Get-Content -LiteralPath (Join-Path $job.root 'report.json') -Raw|ConvertFrom-Json
    $result=$capture.results[0]
    if(!$result.goal_stop){continue}
    $rows=@(Get-Content -LiteralPath (Join-Path $result.root 'bot.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
    $serverLog=Join-Path $result.root 'server.log'
    $release=@(Get-Content -LiteralPath $serverLog|Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
    if($release.Count -ne 1){throw 'Missing release'}
    $context=@{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value}
    $events=@(Read-DamageEvents $serverLog)
    $receipt=Get-CombatEarliestGoalReceipt $rows $events $context $result.goal_stop.classes $result.first_life.map
    if(!$receipt -or $receipt.observed_frame -ne $receipt.kill_frame+1){throw 'No earliest complete terminal on real trace'}
    $checks+=@{seed=$job.seed;variant=$job.plan_index;old_observed_frame=$result.goal_stop.observed_frame;new_observed_frame=$receipt.observed_frame;kill_frame=$receipt.kill_frame}
}
$matched=@($checks|Where-Object seed -eq 1252026752)
if($matched.Count -ne 2 -or @($matched.new_observed_frame|Sort-Object -Unique).Count -ne 1 -or $matched[0].new_observed_frame -ne 231){throw 'Real failing pair not repaired'}
$null=New-Item -ItemType Directory -Path $Out -ErrorAction Stop
@{state='offline_real_trace_verified';checks=$checks;training_performed=$false;native_repeat_pending=$true;
    script_sha256=(Get-FileHash -LiteralPath "$PSScriptRoot/combat_goal_stop.ps1" -Algorithm SHA256).Hash.ToLower();
    test_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLower()}|
    ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $Out 'report.json') -Encoding utf8NoBOM
$checks|ConvertTo-Json -Compress
