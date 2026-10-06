[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Checkpoint,
    [Parameter(Mandatory)][string[]]$RetentionRollout,
    [Parameter(Mandatory)][string[]]$InputRollout,
    [string]$ReferenceModel='',
    [ValidateRange(1,8)][int]$Iterations=4,
    [ValidateRange(1,10000)][int]$Epochs=2000,
    [int]$Seed=18600,[int]$MixedEvalSeed=18700,[int]$SoloEvalSeed=18800,
    [ValidateRange(1024,65530)][int]$Port=33100,
    [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',
    [string]$RewardConfig='',
    [Parameter(Mandatory)][string]$OutputRoot
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$RewardConfig){$RewardConfig=Join-Path $PSScriptRoot 'scenarios/combat-reward-maneuver-v4.json'}
if(!$ReferenceModel){$ReferenceModel=$Model}
foreach($start in @($Seed,$MixedEvalSeed,$SoloEvalSeed)){
    if($start -lt 0 -or [long]$start+4*$Iterations-1 -gt 2147483647){throw 'Valid independent seeds required'}
}
$trainSeeds=@($Seed..($Seed+4*$Iterations-1))
$mixedSeeds=@($MixedEvalSeed..($MixedEvalSeed+3));$soloSeeds=@($SoloEvalSeed..($SoloEvalSeed+3))
if(@($mixedSeeds+$soloSeeds|Where-Object {$_ -in $trainSeeds}).Count -or @($mixedSeeds|Where-Object {$_ -in $soloSeeds}).Count){throw 'Training and evaluation seeds overlap'}
$Model=(Resolve-Path -LiteralPath $Model).Path;$Checkpoint=(Resolve-Path -LiteralPath $Checkpoint).Path
$ReferenceModel=(Resolve-Path -LiteralPath $ReferenceModel).Path
$Python=(Resolve-Path -LiteralPath $Python).Path
$RetentionRollout=@($RetentionRollout|ForEach-Object{(Resolve-Path -LiteralPath $_).Path})
$InputRollout=@($InputRollout|ForEach-Object{(Resolve-Path -LiteralPath $_).Path})
foreach($data in $RetentionRollout+$InputRollout){
    foreach($line in Get-Content (Join-Path $data 'rollout.jsonl')){
        $seenSeed=($line|ConvertFrom-Json).seed
        if($seenSeed -in $trainSeeds+$mixedSeeds+$soloSeeds){throw 'Existing training observations overlap planned seeds'}
    }
}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$frozenReward=Join-Path $OutputRoot 'reward-config.json';Copy-Item -LiteralPath $RewardConfig -Destination $frozenReward
$rewardSHA=(Get-FileHash -LiteralPath $frozenReward).Hash
$trainer=Join-Path $PSScriptRoot 'fork_combat_group.py';$trainerSHA=(Get-FileHash -LiteralPath $trainer).Hash
$diagnoser=Join-Path $PSScriptRoot 'diagnose_combat.py';$diagnoserSHA=(Get-FileHash -LiteralPath $diagnoser).Hash
$initialSHA=(Get-FileHash -LiteralPath $Model).Hash
$referenceSHA=(Get-FileHash -LiteralPath $ReferenceModel).Hash
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache';$env:GOTOOLCHAIN='auto'
$dataTool=Join-Path $OutputRoot 'q2ppo-data.exe'
Push-Location $repo
try{
    go build -o $dataTool ./cmd/q2ppo-data;if($LASTEXITCODE){throw 'Native proof data tool build failed'}
    $collected=@();$progress=@()
    for($iteration=1;$iteration -le $Iterations;$iteration++){
        if((Get-FileHash -LiteralPath $trainer).Hash -ne $trainerSHA -or (Get-FileHash -LiteralPath $frozenReward).Hash -ne $rewardSHA){throw 'Frozen training inputs changed'}
        $dir=Join-Path $OutputRoot "iteration-$iteration";New-Item -ItemType Directory -Path $dir|Out-Null
        $batch=Join-Path $dir 'batch';$data=Join-Path $dir 'rollout';$update=Join-Path $dir 'update'
        & "$PSScriptRoot/run_learned_combat_baseline.ps1" -Workers 4 -EpisodesPerWorker 1 -Timescale 2 -GameFrames 300 -Mixed -Loadout blaster -CombatMode learned -ProviderFile $Model -Synchronous -RewardConfig $frozenReward -Seed ($Seed+4*($iteration-1)) -Port $Port -OutputRoot $batch
        & $dataTool --batch $batch --model $Model --out $data;if($LASTEXITCODE){throw 'Native replay rejected curriculum data'}
        $collected+=$data
        $fitArgs=@($trainer,'--model',$Model,'--checkpoint',$Checkpoint,'--epochs',"$Epochs",'--out',$update)
        foreach($rollout in $InputRollout+$collected){$fitArgs+=@('--rollout',$rollout)}
        foreach($rollout in $RetentionRollout){$fitArgs+=@('--retain-rollout',$rollout)}
        & $Python @fitArgs;if($LASTEXITCODE){throw 'Supervised group fit failed'}
        $progress+=@{iteration=$iteration;seed=$Seed+4*($iteration-1);batch=$batch;rollout=$data;update=$update}
        $progress|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'progress.json') -Encoding utf8NoBOM
        $Model=Join-Path $update 'weights.json';$Checkpoint=Join-Path $update 'checkpoint.pt'
    }
    foreach($fixture in @('mixed','solo')){
        foreach($phase in @('before','after')){
            $inputModel=if($phase -eq 'before'){$ReferenceModel}else{$Model}
            $evalModel=Join-Path $OutputRoot "$fixture-$phase.json"
            $m=Get-Content -LiteralPath $inputModel -Raw|ConvertFrom-Json;$m.deterministic=$true
            $m|ConvertTo-Json -Depth 10 -Compress|Set-Content $evalModel -Encoding utf8NoBOM
            $runArgs=@{Workers=4;EpisodesPerWorker=1;Timescale=2;GameFrames=300;Loadout='blaster';CombatMode='learned';ProviderFile=$evalModel;Synchronous=$true;RewardConfig=$frozenReward;Port=$Port;OutputRoot=(Join-Path $OutputRoot "$fixture/evaluation-$phase")}
            if($fixture -eq 'mixed'){$runArgs.Mixed=$true;$runArgs.Seed=$MixedEvalSeed}else{$runArgs.ReleaseGameFrame=100;$runArgs.Seed=$SoloEvalSeed}
            & "$PSScriptRoot/run_learned_combat_baseline.ps1" @runArgs
        }
        $manifests=@('before','after'|ForEach-Object{Get-Content (Join-Path $OutputRoot "$fixture/evaluation-$_/manifest.json") -Raw|ConvertFrom-Json})
        foreach($field in @('release_game_frame','post_frame_rng_reset','source_fingerprint','native_source_fingerprint','game_frames','timescale','loadout','mixed','health_kit','synchronous','reward_config_sha256')){
            if($manifests[0].$field -ne $manifests[1].$field){throw "Evaluation mismatch $field"}
        }
        foreach($phase in @('before','after')){
            $r=Get-Content (Join-Path $OutputRoot "$fixture/evaluation-$phase/report.json") -Raw|ConvertFrom-Json
            if(!$r.capture_complete -or !$r.provenance_valid -or $r.results.Count -ne 4){throw 'Incomplete evaluation'}
            foreach($e in $r.results){if(!$e.capture_valid -or !$e.dispatch_valid -or !$e.seed_confirmed){throw 'Invalid evaluation episode'}}
        }
        if((Get-FileHash -LiteralPath $diagnoser).Hash -ne $diagnoserSHA){throw 'Diagnostics changed'}
        & $Python $diagnoser --batch (Join-Path $OutputRoot "$fixture/evaluation-before") --batch (Join-Path $OutputRoot "$fixture/evaluation-after") --out (Join-Path $OutputRoot "$fixture/diagnostics.json")
        if($LASTEXITCODE){throw 'Diagnostics failed'}
    }
    if((Get-FileHash -LiteralPath $ReferenceModel).Hash -ne $referenceSHA){throw 'Reference changed'}
    @{version='combat_observed_group_curriculum_v1';iterations=$Iterations;epochs=$Epochs;initial_weights_sha256=$initialSHA;reference_weights_sha256=$referenceSHA;final_weights_sha256=(Get-FileHash -LiteralPath $Model).Hash;training_seeds=$trainSeeds;mixed_eval_seeds=$mixedSeeds;solo_eval_seeds=$soloSeeds;progress=$progress;scope='Fixed-budget offline group movement/aim supervision on verified old training and fresh policy states, not PPO updates. Four independent instances x2 per batch. Solo retention observations exclude evaluation. No best checkpoint selection or live promotion.'}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8NoBOM
}finally{Pop-Location}
