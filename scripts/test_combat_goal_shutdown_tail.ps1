param([Parameter(Mandatory)][string]$CaptureRoot,[Parameter(Mandatory)][string]$Out)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/combat_goal_stop.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
$root=(Resolve-Path -LiteralPath $CaptureRoot).Path
if(Test-Path -LiteralPath $Out){throw 'Fresh output required'}
$report=Get-Content -LiteralPath (Join-Path $root 'report.json') -Raw|ConvertFrom-Json
if($report.capture_complete -or !$report.provenance_valid -or $report.results.Count -ne 1 -or $report.results[0].error -ne 'Unverified goal stop'){throw 'Expected original goal-validation failure'}
$worker=$report.results[0].root
$paths=@('bot.jsonl','server.log','goal-stop.json','goal.stop')|ForEach-Object {Join-Path $worker $_}
$pins=@{};foreach($p in $paths){$pins[$p]=(Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant()}
$rows=@(Get-Content -LiteralPath (Join-Path $worker 'bot.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
$events=@(Read-DamageEvents (Join-Path $worker 'server.log'))
$saved=Get-Content -LiteralPath (Join-Path $worker 'goal-stop.json') -Raw|ConvertFrom-Json
$release=@(Get-Content -LiteralPath (Join-Path $worker 'server.log')|Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
if($release.Count -ne 1){throw 'Unique native release required'}
$context=@{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value}
$map=$rows[0].map
if(Get-CombatGoalReceipt $rows[-1] $events $context $saved.classes $map){throw 'Fixture no longer reproduces last-row failure'}
$verified=Get-CombatVerifiedSavedGoalReceipt $saved $rows $events $context $saved.classes $map
if(!$verified -or $verified.kill_frame -ne 261 -or $verified.observed_frame -ne 262 -or $verified.health -ne 11){throw 'Original complete goal not independently verified'}
$deaths=@($events|Where-Object {$_.target -eq $verified.actor -and $_.killed -and $_.target_class -eq 'player'})
if($deaths.Count -ne 1 -or $deaths[0].frame -ne 263){throw 'Expected post-goal native death missing'}
$falseReceipt=$saved.PSObject.Copy();$falseReceipt.observed_frame=264
if(Get-CombatVerifiedSavedGoalReceipt $falseReceipt $rows $events $context $saved.classes $map){throw 'Late receipt accepted'}
$null=New-Item -ItemType Directory -Path $Out
foreach($name in @('bot.jsonl','server.log')){
    $null=New-Item -ItemType HardLink -Path (Join-Path $Out $name) -Target (Join-Path $worker $name)
}
$null=& "$PSScriptRoot/analyze_combat_first_life.ps1" -RunRoot $Out -EndObservationFrame $verified.observed_frame
$life=Get-Content -LiteralPath (Join-Path $Out 'combat-first-life.json') -Raw|ConvertFrom-Json
if($life.end_frame -ne 262 -or $life.minimum_health -ne 11 -or $life.end_reason -ne 'observed_goal_boundary' -or $life.damage.received_health_damage -ne 89){throw 'First-life metrics include post-goal death'}
foreach($p in $pins.Keys){if((Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant() -ne $pins[$p]){throw 'Original evidence changed'}}
@{state='complete';version='combat_goal_shutdown_tail_regression_v1';goal=$verified;post_goal_death_frame=$deaths[0].frame;
    received_before_goal=$life.damage.received_health_damage;original_evidence_unchanged=$true;inputs=$pins;
    neural_execution=$false;new_games=$false;training_performed=$false;promotion=$false}|
    ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $Out 'report.json') -Encoding utf8NoBOM
@{state='complete';report_sha256=(Get-FileHash -LiteralPath (Join-Path $Out 'report.json')).Hash.ToLowerInvariant()}|
    ConvertTo-Json|Set-Content -LiteralPath (Join-Path $Out 'complete.json') -Encoding utf8NoBOM
'Real shutdown-tail regression passed; original trace unchanged'
