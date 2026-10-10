[CmdletBinding()]
param([string]$Plan,[int]$TaskIndex,[int]$SeedIndex,[string]$Mode,[int]$Port,[string]$OutputRoot,[string]$BinaryBundle='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$schedule=Get-Content -LiteralPath $Plan -Raw|ConvertFrom-Json
$task=$schedule.tasks[$TaskIndex];$ep=$task.episode;$seed=[int]$task.seeds[$SeedIndex]
if($ep.recipe.runner -ne 'combat-baseline' -or $Mode -notin $task.modes){throw 'Unsupported single episode job'}
$argsMap=@{OutputRoot=$OutputRoot;Workers=1;EpisodesPerWorker=1;Timescale=2;GameFrames=$ep.game_frames;Seed=$seed;Port=$Port;Skill=$ep.skill;CombatMode=$Mode;Loadout=$ep.recipe.loadout;Synchronous=$true;StopOnGoal=$true;StopOnFirstDeath=$true;ReleaseGameFrame=100;RewardConfig=(Join-Path $repo $ep.recipe.reward_config)}
if($BinaryBundle){$argsMap.BinaryBundle=$BinaryBundle}
if($Mode -eq 'learned'){
    if((Get-FileHash $schedule.model_path).Hash.ToLowerInvariant() -ne $schedule.model_sha256){throw 'Frozen model changed'}
    $argsMap.ProviderFile=$schedule.model_path
}
if($ep.recipe.mixed){$argsMap.Mixed=$true}
if($task.instances.Count){
    $instance=$task.instances[$SeedIndex]
    if($instance.engine_seed -ne $seed){throw 'Fixture seed differs'}
    $fixtureFile="$OutputRoot-fixture.json"
    @{instances=@($instance)}|ConvertTo-Json -Depth 18|Set-Content -LiteralPath $fixtureFile -Encoding utf8NoBOM
    $argsMap.GeneratedFixtures=$fixtureFile
}
& "$PSScriptRoot/run_learned_combat_baseline.ps1" @argsMap
$r=Get-Content -LiteralPath "$OutputRoot/report.json" -Raw|ConvertFrom-Json
if(!$r.capture_complete -or !$r.provenance_valid -or $r.results.Count -ne 1 -or $r.results[0].seed -ne $seed -or !$r.results[0].capture_valid){throw 'Single episode native proof failed'}
