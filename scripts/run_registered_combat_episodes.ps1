[CmdletBinding()]
param([Parameter(Mandatory)][string]$Plan,[int]$Port=34300,[switch]$DryRun)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$planPath=(Resolve-Path -LiteralPath $Plan).Path
$schedule=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json
if($schedule.version -ne 1 -or $schedule.workers -ne 4 -or !$schedule.tasks.Count){throw 'Invalid registered schedule'}
if((Get-FileHash $schedule.registry_path).Hash.ToLowerInvariant() -ne $schedule.registry_sha256){throw 'Registry index changed'}
Push-Location $repo
try{
    & go run ./cmd/q2episode --verify-plan $planPath --root $repo
    if($LASTEXITCODE){throw 'Frozen generated plan verification failed'}
    $listing=& go run ./cmd/q2episode --registry $schedule.registry_path --list
    if($LASTEXITCODE){throw 'Registry validation failed'}
    $catalog=($listing -join "`n")|ConvertFrom-Json
}finally{Pop-Location}
if($schedule.model_path -and (Get-FileHash $schedule.model_path).Hash.ToLowerInvariant() -ne $schedule.model_sha256){throw 'Frozen model changed'}
foreach($task in $schedule.tasks){
    $entry=@($catalog.episodes|Where-Object id -eq $task.episode.id)
    if($entry.Count -ne 1 -or $entry[0].status -ne 'runnable' -or $catalog.episode_hashes.($task.episode.id) -ne $task.episode_sha256){throw 'Episode changed or is not runnable'}
    $task.episode=$entry[0]
    $range=$task.episode.splits.($task.split)
    if(!$range -or !$task.seeds.Count -or $task.seeds.Count%4 -or $task.seeds.Count -gt 80){throw 'Invalid episode cohort'}
    for($i=0;$i -lt $task.seeds.Count;$i++){
        $seed=$task.seeds[$i]
        if($seed -lt $range.start -or $seed -ge $range.start+$range.count -or ($i -gt 0 -and $seed -ne $task.seeds[$i-1]+1)){throw 'Seed outside registered split or not contiguous'}
    }
    if($task.split -eq 'train' -and !$task.episode.ppo_trainable){throw 'Evaluation-only episode selected for training'}
    foreach($mode in $task.modes){if($mode -notin $task.episode.modes){throw 'Controller not supported by episode'};if($mode -eq 'learned' -and !$schedule.model_path){throw 'Missing model'}}
    $expected=Join-Path $PSScriptRoot $(if($task.episode.recipe.runner -eq 'combat-baseline'){'run_learned_combat_baseline.ps1'}else{'run_learned_campaign_comparison.ps1'})
    if([IO.Path]::GetFullPath($task.runner_path) -ne [IO.Path]::GetFullPath($expected) -or (Get-FileHash $expected).Hash.ToLowerInvariant() -ne $task.runner_sha256){throw 'Runner binding changed'}
    if($task.episode.recipe.reward_config -and (Get-FileHash (Join-Path $repo $task.episode.recipe.reward_config)).Hash.ToLowerInvariant() -ne $task.reward_sha256){throw 'Reward changed'}
}
if($DryRun){Write-Output "Verified plan: $planPath";return}
if(Test-Path -LiteralPath $schedule.output_root){throw 'Fresh run output required'}
New-Item -ItemType Directory -Path $schedule.output_root|Out-Null
Copy-Item -LiteralPath $planPath -Destination (Join-Path $schedule.output_root 'plan.json')
$records=@()
try{
    foreach($task in $schedule.tasks){foreach($mode in $task.modes){
        $ep=$task.episode
        $out=Join-Path $schedule.output_root "$($ep.id)-$mode"
        $argsMap=@{OutputRoot=$out;Workers=4;Timescale=2;GameFrames=$ep.game_frames;Seed=$task.seeds[0];Port=$Port;Skill=$ep.skill;CombatMode=$mode;Loadout=$ep.recipe.loadout;EpisodesPerWorker=[int]($task.seeds.Count/4);Synchronous=$true;StopOnGoal=$true;StopOnFirstDeath=$true;ReleaseGameFrame=100;RewardConfig=(Join-Path $repo $ep.recipe.reward_config)}
        if($mode -eq 'learned'){$argsMap.ProviderFile=$schedule.model_path}
        if($null -ne $schedule.policy_sampling_seed_offset){$argsMap.PolicySamplingSeedOffset=[long]$schedule.policy_sampling_seed_offset}
        if($ep.recipe.mixed){$argsMap.Mixed=$true}
        if($task.instances.Count){
            $fixtureInput=Join-Path $schedule.output_root "$($ep.id)-$mode-fixtures.json"
            @{instances=$task.instances}|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $fixtureInput -Encoding utf8NoBOM
            $argsMap.GeneratedFixtures=$fixtureInput
        }
        if($ep.recipe.runner -eq 'combat-baseline'){
            & "$PSScriptRoot/run_learned_combat_baseline.ps1" @argsMap
        }else{
            $campaign=@{OutputRoot=$out;Workers=4;GameFrames=$ep.game_frames;Seed=($task.seeds[0]-$(if($ep.map -eq 'base2'){100}else{0}));Port=$Port;Skill=$ep.skill;Maps=@($ep.map);Modes=@($mode);EpisodesPerMap=$task.seeds.Count}
            if($mode -eq 'learned'){$campaign.Model=$schedule.model_path}
            & "$PSScriptRoot/run_learned_campaign_comparison.ps1" @campaign
        }
        $report=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json
        if((($report.results.seed|Sort-Object) -join ',') -ne (($task.seeds|Sort-Object) -join ',')){throw 'Native seed allocation differs from registry'}
        if($ep.recipe.runner -eq 'combat-baseline'){
            if(!$report.capture_complete -or !$report.provenance_valid){throw 'Capture verification failed'}
        }elseif(!$report.source_unchanged -or @($report.results|Where-Object {!$_.infrastructure_ok}).Count){throw 'Campaign verification failed'}
        $record=@{episode_id=$ep.id;episode_revision=$ep.revision;episode_sha256=$task.episode_sha256;split=$task.split;mode=$mode;seeds=$task.seeds;artifacts=$out;report_sha256=(Get-FileHash (Join-Path $out 'report.json')).Hash.ToLowerInvariant();model_sha256=$(if($mode -eq 'learned'){$schedule.model_sha256}else{''});capture_verified=$true}
        $record|ConvertTo-Json -Depth 7|Set-Content (Join-Path $out 'registry-binding.json') -Encoding utf8
        $records+=$record
    }}
    @{state='complete';records=$records}|ConvertTo-Json -Depth 9|Set-Content (Join-Path $schedule.output_root 'report.json') -Encoding utf8
}catch{
    @{state='failed';reason=$_.Exception.Message;records=$records}|ConvertTo-Json -Depth 9|Set-Content (Join-Path $schedule.output_root 'report.json') -Encoding utf8
    throw
}
