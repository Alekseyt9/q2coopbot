[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Checkpoint,
    [ValidateRange(1,20)][int]$Iterations=4,
    [ValidateRange(1,20)][int]$EpisodesPerWorker=1,
    [int]$Seed=13600,[int]$EvalSeed=13700,
    [ValidateSet(0,10,20,30,40,60,100)][int]$TrainingMonsterHealth=0,
    [ValidateSet(0,100)][int]$ReleaseGameFrame=0,
    [switch]$Mixed,
    [ValidateSet('blaster','machinegun')][string]$Loadout='blaster',
    [ValidateRange(0,3)][int]$Skill=1,
    [ValidateSet('uniform','mixed-solo-mixed')][string]$TrainingPattern='uniform',
    [ValidateSet('standard','remaining-far')][string]$TrainingSoloFixture='standard',
    [ValidateRange(20,500)][int]$GameFrames=300,
    [ValidateRange(1024,65530)][int]$Port=33100,
    [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',
    [string]$Config='',[string]$RewardConfig='',[string]$OutputRoot='',
    [string]$AnchorModel='',[double]$RetentionWeight=1.,[ValidateRange(2,100)][int]$RetentionUpdates=4,
    [ValidateSet('linear','constant')][string]$RetentionMode='linear',
    [string]$InitialBatch='',
    [string]$RetentionBank='',[double]$BankWeight=1.
)
$ErrorActionPreference='Stop'
if($RetentionMode -eq 'constant' -and !$AnchorModel){throw 'Constant retention requires an anchor before collection'}
if($Mixed -and $TrainingMonsterHealth -ne 0){throw 'Mixed training requires stock monster health'}
if($TrainingPattern -ne 'uniform' -and (!$Mixed -or $EpisodesPerWorker -ne 3 -or $TrainingMonsterHealth)){throw 'Mixed/Solo training requires Mixed and three episodes per worker without health overrides'}
if($TrainingSoloFixture -ne 'standard' -and ($TrainingMonsterHealth -or ($Mixed -and $TrainingPattern -eq 'uniform'))){throw 'Remaining-Parasite training requires Solo episodes without health overrides'}
if($Loadout -eq 'machinegun' -and ($TrainingMonsterHealth -or $TrainingPattern -ne 'uniform' -or $TrainingSoloFixture -ne 'standard')){throw 'Machinegun experiment keeps stock health and uniform fixtures'}
$repo=Split-Path $PSScriptRoot -Parent
if(!$Config){$Config=Join-Path $PSScriptRoot 'scenarios/combat-ppo-v1.json'}
if(!$RewardConfig){$RewardConfig=Join-Path $PSScriptRoot 'scenarios/combat-reward-v1.json'}
$RewardConfig=(Resolve-Path -LiteralPath $RewardConfig).Path
$Model=(Resolve-Path -LiteralPath $Model).Path;$Checkpoint=(Resolve-Path -LiteralPath $Checkpoint).Path
$Python=(Resolve-Path -LiteralPath $Python).Path;$Config=(Resolve-Path -LiteralPath $Config).Path
if($AnchorModel){$AnchorModel=(Resolve-Path -LiteralPath $AnchorModel).Path;$anchorSHA=(Get-FileHash $AnchorModel).Hash}
$trainingObjective=(Get-Content -LiteralPath $Config -Raw|ConvertFrom-Json).objective_reward_sha256
$rewardVersion=(Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json).version
if(($rewardVersion -in @('combat_reward_v2','combat_reward_v3','combat_reward_v4','combat_reward_v5') -and !$trainingObjective) -or ($trainingObjective -and $trainingObjective -ne (Get-FileHash -LiteralPath $RewardConfig).Hash)){throw 'Training objective must match reward config before collection'}
if($rewardVersion -in @('combat_reward_v3','combat_reward_v4','combat_reward_v5') -and (Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json).aim_gamma -ne (Get-Content -LiteralPath $Config -Raw|ConvertFrom-Json).gamma){throw 'Shaping gamma must match PPO before collection'}
$batchEpisodes=4*$EpisodesPerWorker
if($Seed -lt 0 -or [long]$Seed+$batchEpisodes*$Iterations-1 -gt 2147483647 -or $EvalSeed -lt 0 -or [long]$EvalSeed+3 -gt 2147483647){throw 'Valid independent seeds required'}
if($EvalSeed -le $Seed+$batchEpisodes*$Iterations-1 -and $EvalSeed+3 -ge $Seed){throw 'Evaluation seeds overlap training'}
if($RetentionBank){
    if(!$AnchorModel -or [double]::IsNaN($BankWeight) -or [double]::IsInfinity($BankWeight) -or $BankWeight -le 0){throw 'Bank requires pinned anchor and positive finite weight'}
    $RetentionBank=(Resolve-Path -LiteralPath $RetentionBank).Path;$bankSHA=(Get-FileHash $RetentionBank).Hash
    & $Python (Join-Path $PSScriptRoot 'combat_retention_bank.py') --validate $RetentionBank --anchor-model $AnchorModel --ppo-seed $Seed --eval-seed $EvalSeed --iterations $Iterations --episodes-per-worker $EpisodesPerWorker
    if($LASTEXITCODE){throw 'Bank validation failed before collection'}
}
if($InitialBatch){
    $InitialBatch=(Resolve-Path -LiteralPath $InitialBatch).Path
    $initialManifestPath=Join-Path $InitialBatch 'manifest.json';$initialReportPath=Join-Path $InitialBatch 'report.json'
    $initialManifest=Get-Content -LiteralPath $initialManifestPath -Raw|ConvertFrom-Json
    $initialReport=Get-Content -LiteralPath $initialReportPath -Raw|ConvertFrom-Json
    $expected=@{model_weights_sha256=(Get-FileHash $Model).Hash;reward_config_sha256=(Get-FileHash $RewardConfig).Hash;training_monster_health=$TrainingMonsterHealth;release_game_frame=$ReleaseGameFrame;timescale=2;game_frames=$GameFrames;workers=4;episodes_per_worker=$EpisodesPerWorker;loadout=$Loadout;mixed=[bool]$Mixed;synchronous=$true;remote_peer=$false;feedback=$false;health_kit=$false}
    if(!$initialManifest.episode_pattern){$initialPattern='uniform'}else{$initialPattern=$initialManifest.episode_pattern}
    $initialSkill=if($null -ne $initialManifest.skill){$initialManifest.skill}else{1}
    if($initialSkill -ne $Skill){throw 'Initial batch mismatch: skill'}
    if($initialPattern -ne $TrainingPattern){throw 'Initial batch mismatch: episode_pattern'}
    $initialFixture=if($initialManifest.solo_fixture){$initialManifest.solo_fixture}else{'standard'}
    if($initialFixture -ne $TrainingSoloFixture){throw 'Initial batch mismatch: solo_fixture'}
    foreach($field in $expected.Keys){if($initialManifest.$field -ne $expected[$field]){throw "Initial batch mismatch: $field"}}
    if(!$initialReport.capture_complete -or !$initialReport.provenance_valid -or $initialReport.completed_episodes -ne $batchEpisodes){throw 'Initial batch incomplete'}
    $actualSeeds=@($initialReport.results|Sort-Object seed|ForEach-Object{if(!$_.seed_confirmed -or !$_.dispatch_valid){throw 'Initial batch seed/dispatch invalid'};[int]$_.seed})
    if(($actualSeeds -join ',') -ne (($Seed..($Seed+$batchEpisodes-1)) -join ',')){throw 'Initial batch seeds mismatch'}
    $initialManifestSHA=(Get-FileHash $initialManifestPath).Hash;$initialReportSHA=(Get-FileHash $initialReportPath).Hash
}
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/combat-ppo-cycle-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null;$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$frozen=Join-Path $OutputRoot 'config.json';Copy-Item -LiteralPath $Config -Destination $frozen
$frozenReward=Join-Path $OutputRoot 'reward-config.json';Copy-Item -LiteralPath $RewardConfig -Destination $frozenReward
$rewardSHA=(Get-FileHash -LiteralPath $frozenReward).Hash
$configSHA=(Get-FileHash -LiteralPath $frozen).Hash;$trainer=Join-Path $PSScriptRoot 'ppo_combat.py';$trainerSHA=(Get-FileHash -LiteralPath $trainer).Hash
Copy-Item -LiteralPath $trainer -Destination (Join-Path $OutputRoot 'trainer.py')
$bankHelper=Join-Path $PSScriptRoot 'combat_retention_bank.py';$bankHelperSHA=(Get-FileHash $bankHelper).Hash
Copy-Item -LiteralPath $bankHelper -Destination (Join-Path $OutputRoot 'bank-helper.py')
$diagnoser=Join-Path $PSScriptRoot 'diagnose_combat.py';$diagnoserSHA=(Get-FileHash -LiteralPath $diagnoser).Hash
Copy-Item -LiteralPath $diagnoser -Destination (Join-Path $OutputRoot 'diagnose_combat.py')
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache';$env:GOTOOLCHAIN='auto'
$dataTool=Join-Path $OutputRoot 'q2ppo-data.exe'
Push-Location $repo
try{
    go build -o $dataTool ./cmd/q2ppo-data;if($LASTEXITCODE){throw 'PPO data tool build failed'}
    $initialModel=$Model;$initialCheckpoint=$Checkpoint;$steps=@()
    for($iteration=1;$iteration -le $Iterations;$iteration++){
        if((Get-FileHash $frozen).Hash -ne $configSHA -or (Get-FileHash $trainer).Hash -ne $trainerSHA -or (Get-FileHash $frozenReward).Hash -ne $rewardSHA){throw 'Frozen training inputs changed'}
        if((Get-FileHash $bankHelper).Hash -ne $bankHelperSHA){throw 'Frozen bank helper changed'}
        $dir=Join-Path $OutputRoot "iteration-$iteration";New-Item -ItemType Directory -Path $dir|Out-Null
        $batch=Join-Path $dir 'batch';$data=Join-Path $dir 'rollout';$update=Join-Path $dir 'update'
        if($iteration -eq 1 -and $InitialBatch){
            if((Get-FileHash $initialManifestPath).Hash -ne $initialManifestSHA -or (Get-FileHash $initialReportPath).Hash -ne $initialReportSHA){throw 'Initial batch changed'}
            $batch=$InitialBatch
            @{batch=$batch;manifest_sha256=$initialManifestSHA;report_sha256=$initialReportSHA;scope='Shared first on-policy batch only; later batches collected from each own resumed policy'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $dir 'reused-batch.json') -Encoding utf8NoBOM
        }else{
            & "$PSScriptRoot/run_learned_combat_baseline.ps1" -Workers 4 -EpisodesPerWorker $EpisodesPerWorker -Timescale 2 -GameFrames $GameFrames -ReleaseGameFrame $ReleaseGameFrame -Mixed:$Mixed -EpisodePattern $TrainingPattern -SoloFixture $TrainingSoloFixture -Loadout $Loadout -Skill $Skill -CombatMode learned -ProviderFile $Model -Synchronous -RewardConfig $frozenReward -TrainingMonsterHealth $TrainingMonsterHealth -Seed ($Seed+$batchEpisodes*($iteration-1)) -Port $Port -OutputRoot $batch
        }
        & $dataTool --batch $batch --model $Model --out $data;if($LASTEXITCODE){throw "Iteration $iteration native replay rejected"}
        $updateArgs=@($trainer,'--model',$Model,'--resume',$Checkpoint,'--data',$data,'--config',$frozen,'--out',$update)
        if($AnchorModel){if((Get-FileHash $AnchorModel).Hash -ne $anchorSHA){throw 'Pinned anchor changed'};$updateArgs+=@('--anchor-model',$AnchorModel,'--retention-weight',$RetentionWeight.ToString([cultureinfo]::InvariantCulture),'--retention-updates',"$RetentionUpdates",'--retention-mode',$RetentionMode)}
        if($RetentionBank){if((Get-FileHash $RetentionBank).Hash -ne $bankSHA){throw 'Pinned bank changed'};$updateArgs+=@('--retention-bank',$RetentionBank,'--bank-weight',$BankWeight.ToString([cultureinfo]::InvariantCulture))}
        & $Python @updateArgs
        if($LASTEXITCODE){throw "Iteration $iteration PPO update failed"}
        $report=Get-Content -LiteralPath (Join-Path $update 'report.json') -Raw|ConvertFrom-Json
        if(!$report.resume_sha256 -or $report.final_approx_kl -gt 0.010001){throw 'Update acceptance failed'}
        $steps+=@{iteration=$iteration;seed=$Seed+$batchEpisodes*($iteration-1);batch=$batch;rollout=$data;update=$update;report=$report}
        $Model=Join-Path $update 'weights.json';$Checkpoint=Join-Path $update 'checkpoint.pt'
        $steps|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $OutputRoot 'progress.json') -Encoding utf8NoBOM
    }
    foreach($name in @('before','after')){
        if((Get-FileHash $frozenReward).Hash -ne $rewardSHA){throw 'Frozen reward changed'}
        $inputModel=if($name -eq 'before'){$initialModel}else{$Model}
        $evalModel=Join-Path $OutputRoot "$name.json";$m=Get-Content $inputModel -Raw|ConvertFrom-Json;$m.deterministic=$true
        $m|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $evalModel -Encoding utf8NoBOM
        & "$PSScriptRoot/run_learned_combat_baseline.ps1" -Workers 4 -EpisodesPerWorker 1 -Timescale 2 -GameFrames $GameFrames -ReleaseGameFrame $ReleaseGameFrame -Mixed:$Mixed -Loadout $Loadout -Skill $Skill -CombatMode learned -ProviderFile $evalModel -Synchronous -RewardConfig $frozenReward -Seed $EvalSeed -Port $Port -OutputRoot (Join-Path $OutputRoot "evaluation-$name")
    }
    $reports=@('before','after'|ForEach-Object{Get-Content (Join-Path $OutputRoot "evaluation-$_/report.json") -Raw|ConvertFrom-Json})
    $manifests=@('before','after'|ForEach-Object{Get-Content (Join-Path $OutputRoot "evaluation-$_/manifest.json") -Raw|ConvertFrom-Json})
    foreach($field in @('release_game_frame','post_frame_rng_reset','source_fingerprint','native_source_fingerprint','game_frames','timescale','loadout','skill','mixed','health_kit','synchronous','reward_config_sha256')){if($manifests[0].$field -ne $manifests[1].$field){throw "Evaluation mismatch $field"}}
    $pairs=@(foreach($s in $EvalSeed..($EvalSeed+3)){
        $metrics=@(foreach($r in $reports){if(!$r.capture_complete -or !$r.provenance_valid){throw 'Invalid evaluation'}
            $episode=@($r.results|Where-Object seed -eq $s);if($episode.Count -ne 1 -or !$episode[0].dispatch_valid -or !$episode[0].seed_confirmed){throw 'Invalid seed pair'}
            $e=$episode[0];$life=$e.first_life
            $classKills=@{};foreach($target in $life.outgoing_by_target_class){$classKills[$target.target_class]=[int]$target.kills}
            @{monster_damage=[double](($life.outgoing_by_target_class|Measure-Object health_damage -Sum).Sum);kills=[int](($life.outgoing_by_target_class|Measure-Object kills -Sum).Sum);kills_by_class=$classKills;received_damage=$life.damage.received_health_damage;end=$life.end_reason;reward=$e.dataset.reward_sum;gameplay_accepted=$e.harness_accepted}
        });@{seed=$s;before=$metrics[0];after=$metrics[1]}
    })
    # A candidate checkpoint continues training even when gameplay fails. No live promotion.
    # Legacy harness acceptance checks rules-specific Shotgun switching/aim ownership.
    # Keep that metric in the report, but use native first-life kills/death for this
    # fixed-Blaster fixture. Capture/dispatch/seed checks have already passed above.
    $eligible=@($pairs|Where-Object {$_.after.kills -lt 1 -or $_.after.end -eq 'first_observed_death'}).Count -eq 0
    if($Mixed){$eligible=$eligible -and @($pairs|Where-Object {$_.after.kills_by_class['monster_parasite'] -lt 1 -or $_.after.kills_by_class['monster_gunner'] -lt 1}).Count -eq 0}
    if((Get-FileHash -LiteralPath $diagnoser).Hash -ne $diagnoserSHA){throw 'Diagnostics source changed'}
    $diagnostics=Join-Path $OutputRoot 'diagnostics.json'
    & $Python $diagnoser --batch (Join-Path $OutputRoot 'evaluation-before') --batch (Join-Path $OutputRoot 'evaluation-after') --out $diagnostics
    if($LASTEXITCODE){throw 'Evaluation diagnostics failed'}
    $summary=@{version='combat_ppo_cycle_v1';episodes_per_worker=$EpisodesPerWorker;training_pattern=$TrainingPattern;training_solo_fixture=$TrainingSoloFixture;loadout=$Loadout;skill=$Skill;training_episodes_per_batch=$batchEpisodes;mixed=[bool]$Mixed;release_game_frame=$ReleaseGameFrame;training_monster_health=$TrainingMonsterHealth;evaluation_monster_health=175;iterations=$Iterations;initial_model=$initialModel;initial_checkpoint=$initialCheckpoint;final_model=$Model;final_checkpoint=$Checkpoint;config_sha256=$configSHA;trainer_sha256=$trainerSHA;diagnoser_sha256=$diagnoserSHA;diagnostics=$diagnostics;steps=$steps;evaluation=$pairs;fixture_promotion_eligible=$eligible;fixture_criterion='All four paired after captures valid; native first-life kill >=1 and no observed death; Mixed additionally requires a kill of both Parasite and Gunner in each episode. gameplay_accepted is the legacy rules-specific harness metric, not learned-policy acceptance.';scope='Fresh on-policy batches with optimizer/RNG resume; paired deterministic evaluation only, no live promotion or statistical generalization claim'}
    $summary|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $OutputRoot 'report.json') -Encoding utf8NoBOM
    "PPO cycle: $OutputRoot"
}finally{Pop-Location}
