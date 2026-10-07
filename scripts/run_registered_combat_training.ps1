[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config,[Parameter(Mandatory)][string]$OutputRoot,
 [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',[int]$Port=34400,[switch]$DryRun)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$configPath=(Resolve-Path -LiteralPath $Config).Path
$cfg=Get-Content -LiteralPath $configPath -Raw|ConvertFrom-Json
foreach($key in $cfg.PSObject.Properties.Name){if($key -notin @('version','comparison_kind','registry','episodes','evaluation_episodes','models','epochs','episodes_per_scene','evaluation_count','training_seed_offset','training_config','anchor','bank','include_rules')){throw "Unknown training field $key"}}
if($cfg.version -ne 1 -or $cfg.comparison_kind -notin @('architecture','continuation') -or !$cfg.models.Count -or !$cfg.episodes.Count -or !$cfg.evaluation_episodes.Count -or $cfg.epochs -lt 1 -or $cfg.epochs -gt 20 -or $cfg.episodes_per_scene -lt 4 -or $cfg.episodes_per_scene -gt 80 -or $cfg.episodes_per_scene%4 -or $cfg.evaluation_count -lt 4 -or $cfg.evaluation_count%4 -or $cfg.training_seed_offset -lt 0){throw 'Invalid registered training configuration'}
function Resolve-RepoFile([string]$Path){return (Resolve-Path -LiteralPath $(if([IO.Path]::IsPathRooted($Path)){$Path}else{Join-Path $repo $Path})).Path}
$registry=Resolve-RepoFile $cfg.registry
$trainingConfig=Resolve-RepoFile $cfg.training_config
$anchor=Resolve-RepoFile $cfg.anchor;$bank=Resolve-RepoFile $cfg.bank
$snapshotRoot=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path $repo $OutputRoot}))
if(Test-Path -LiteralPath $snapshotRoot){throw 'Fresh experiment root required'}
New-Item -ItemType Directory -Path $snapshotRoot|Out-Null
Copy-Item -LiteralPath $configPath -Destination "$snapshotRoot/config.json"
Push-Location $repo
try{
    $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
    & go build -o "$snapshotRoot/q2episode.exe" ./cmd/q2episode;if($LASTEXITCODE){throw 'Episode compiler build failed'}
    & go build -o "$snapshotRoot/q2ppo-data.exe" ./cmd/q2ppo-data;if($LASTEXITCODE){throw 'Native exporter build failed'}
    New-Item -ItemType Directory -Path "$snapshotRoot/python-sources"|Out-Null
    Get-ChildItem -LiteralPath "$PSScriptRoot" -Filter '*.py' -File|Copy-Item -Destination "$snapshotRoot/python-sources"
    Copy-Item -LiteralPath $trainingConfig -Destination "$snapshotRoot/training-config.json"
    Copy-Item -LiteralPath $anchor -Destination "$snapshotRoot/anchor.json"
    Copy-Item -LiteralPath $bank -Destination "$snapshotRoot/bank.json"
    $trainingConfig="$snapshotRoot/training-config.json";$anchor="$snapshotRoot/anchor.json";$bank="$snapshotRoot/bank.json"
    $modelSpecs=@();$ids=@{};$contract='';$initializationGroup=''
    foreach($m in $cfg.models){
        foreach($key in $m.PSObject.Properties.Name){if($key -notin @('id','model','checkpoint','initialization_group','prior_environment_frames')){throw 'Unknown model field'}}
        if($m.id -cnotmatch '^[a-z0-9][a-z0-9-]*$' -or $ids.ContainsKey($m.id)){throw 'Unique safe model IDs required'};$ids[$m.id]=$true
        $model=Resolve-RepoFile $m.model
        $weights=Get-Content -LiteralPath $model -Raw|ConvertFrom-Json
        if($weights.kind -ne 'combat_ppo_v1' -or $weights.deterministic){throw 'Training requires stochastic PPO weights'}
        $key=if($weights.memory){'gru'}elseif($weights.attention){'attention'}elseif($weights.entity_attention){'entity'}else{'mlp'}
        if($key -eq 'mlp' -and $weights.weapon_head){throw 'MLP weapon-head trainer is not implemented yet; choose the common 8-action curriculum or add its validated adapter'}
        $thisContract="$($weights.feature_version):$($weights.actor[-1].bias.Count):$($weights.weapon_head)"
        if($cfg.comparison_kind -eq 'architecture' -and $contract -and $contract -ne $thisContract){throw 'Architecture comparison requires identical feature/action contracts'};$contract=$thisContract
        if($cfg.comparison_kind -eq 'architecture' -and $m.checkpoint){throw 'Architecture benchmark starts fresh optimizers; use continuation to resume existing checkpoints'}
        if($cfg.comparison_kind -eq 'architecture'){
            if(!$m.initialization_group -or $null -eq $m.prior_environment_frames -or $m.prior_environment_frames -ne 0){throw 'Declare a common initialization group and zero prior environment frames for a fresh architecture benchmark'}
            if($initializationGroup -and $initializationGroup -ne $m.initialization_group){throw 'Architecture initializations belong to different groups'}
            $initializationGroup=$m.initialization_group
        }
        $checkpoint=if($m.checkpoint){Resolve-RepoFile $m.checkpoint}else{''}
        if($checkpoint){
            & $Python -c 'import hashlib,json,sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=True); assert c["weights_sha256"]==hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(), "checkpoint/model mismatch"; assert c["config"]==json.load(open(sys.argv[3],encoding="utf-8-sig")), "resume config mismatch"; assert c.get("retention_weights",[0.,0.])==[0.,0.], "active retention requires a separate adapter"' $checkpoint $model $trainingConfig
            if($LASTEXITCODE){throw 'Resume checkpoint does not match model'}
        }
        $modelSpecs+=@{id=$m.id;model=$model;checkpoint=$checkpoint;architecture=$key;initialization_group=$m.initialization_group;prior_environment_frames=$m.prior_environment_frames;initial_sha256=(Get-FileHash $model).Hash.ToLowerInvariant();checkpoint_sha256=$(if($checkpoint){(Get-FileHash $checkpoint).Hash.ToLowerInvariant()}else{''})}
        # Compile every requested training scene and the last epoch before starting any server.
        foreach($offset in @($cfg.training_seed_offset,($cfg.training_seed_offset+($cfg.epochs-1)*$cfg.episodes_per_scene))|Select-Object -Unique){
            $planPath="$snapshotRoot/preflight-$($m.id)-$offset.json"
            & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.episodes -join ',') --split train --mode learned --model $model --count $cfg.episodes_per_scene --seed-offset $offset --out $planPath --artifacts "$snapshotRoot/preflight-runtime-$($m.id)-$offset"
            if($LASTEXITCODE){throw 'Training curriculum cannot be compiled'}
            & "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan $planPath -DryRun
            $p=Get-Content $planPath -Raw|ConvertFrom-Json
            foreach($task in $p.tasks){if($task.reward_sha256 -ne (Get-Content $trainingConfig -Raw|ConvertFrom-Json).objective_reward_sha256){throw 'Episode reward does not match PPO objective'}}
        }
        $evalPlan="$snapshotRoot/preflight-eval-$($m.id).json"
        & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.evaluation_episodes -join ',') --split validation --mode learned --model $model --count $cfg.evaluation_count --out $evalPlan --artifacts "$snapshotRoot/preflight-eval-runtime-$($m.id)"
        if($LASTEXITCODE){throw 'Evaluation curriculum cannot be compiled'}
    }
    if($cfg.include_rules){
        & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.evaluation_episodes -join ',') --split validation --mode rules --count $cfg.evaluation_count --out "$snapshotRoot/preflight-rules.json" --artifacts "$snapshotRoot/rules"
        if($LASTEXITCODE){throw 'Evaluation suite has no rules baseline'}
    }
    @{version=1;comparison_kind=$cfg.comparison_kind;registry_sha256=(Get-FileHash $registry).Hash.ToLowerInvariant();models=$modelSpecs;episodes=$cfg.episodes;evaluation_episodes=$cfg.evaluation_episodes;epochs=$cfg.epochs;episodes_per_scene=$cfg.episodes_per_scene;workers=4;timescale=2;budget_unit='same allocated episodes and frame caps; actual first-life transitions reported separately';training_config_sha256=(Get-FileHash $trainingConfig).Hash.ToLowerInvariant();trainer_modules=@(Get-ChildItem "$snapshotRoot/python-sources" -File|ForEach-Object {@{file=$_.Name;sha256=(Get-FileHash $_.FullName).Hash.ToLowerInvariant()}})}|ConvertTo-Json -Depth 9|Set-Content "$snapshotRoot/protocol.json" -Encoding utf8
    if($DryRun){@{stage='preflight_complete'}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json";Write-Output $snapshotRoot;return}
    & $Python -c 'import sys;sys.path.insert(0,sys.argv[1]);from train_combat_bc import training_devices;assert training_devices()==["cuda"]' "$snapshotRoot/python-sources"
    if($LASTEXITCODE){throw 'CUDA-only training unavailable'}
    $records=@();$evaluations=@()
    foreach($m in $modelSpecs){
        $folder="$snapshotRoot/$($m.id)";New-Item -ItemType Directory -Path $folder|Out-Null
        Copy-Item -LiteralPath $m.model -Destination "$folder/initial.json";$current="$folder/initial.json"
        $checkpoint='';if($m.checkpoint){Copy-Item -LiteralPath $m.checkpoint -Destination "$folder/initial-checkpoint.pt";$checkpoint="$folder/initial-checkpoint.pt"}
        foreach($epoch in 1..$cfg.epochs){
            $step="$folder/epoch-$epoch";New-Item -ItemType Directory -Path $step|Out-Null
            @{stage='capture';model=$m.id;epoch=$epoch;episodes=$cfg.episodes}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
            & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.episodes -join ',') --split train --mode learned --model $current --count $cfg.episodes_per_scene --seed-offset ($cfg.training_seed_offset+($epoch-1)*$cfg.episodes_per_scene) --out "$step/plan.json" --artifacts "$step/capture"
            if($LASTEXITCODE){throw 'Training plan failed'}
            & "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan "$step/plan.json" -Port $Port
            $exports=@()
            foreach($episodeID in $cfg.episodes){
                $batch="$step/capture/$episodeID-learned"
                $export="$step/rollout-$episodeID"
                & "$snapshotRoot/q2ppo-data.exe" --batch $batch --model $current --out $export
                if($LASTEXITCODE){throw 'Verified on-policy export failed'}
                $exports+=$export
            }
            & "$snapshotRoot/q2ppo-data.exe" --merge ($exports -join ',') --out "$step/rollout"
            if($LASTEXITCODE){throw 'Joint curriculum export failed'}
            $trainer=if($m.architecture -eq 'mlp'){'ppo_combat.py'}else{'ppo_recurrent.py'}
            $argsList=@("$snapshotRoot/python-sources/$trainer",'--model',$current,'--data',"$step/rollout",'--config',$trainingConfig,'--out',"$step/update",'--retention-weight','0','--bank-weight','0')
            if($m.architecture -ne 'mlp'){$argsList+=@('--anchor-model',$anchor,'--retention-bank',$bank)}
            if($checkpoint){$argsList+=@('--resume',$checkpoint)}
            @{stage='cuda-update';model=$m.id;epoch=$epoch;episodes=$cfg.episodes}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
            & $Python @argsList > "$step/update.stdout" 2> "$step/update.stderr"
            if($LASTEXITCODE){throw "CUDA update failed; inspect $step/update.stderr"}
            $update=Get-Content "$step/update/report.json" -Raw|ConvertFrom-Json
            if($update.device -ne 'cuda' -or (Get-FileHash "$step/update/weights.json").Hash.ToLowerInvariant() -ne $update.weights_sha256){throw 'Unverified CUDA update'}
            if(Test-Path "$step/update/complete.json"){
                $seal=Get-Content "$step/update/complete.json" -Raw|ConvertFrom-Json
                foreach($entry in @(@('weights.json','weights_sha256'),@('checkpoint.pt','checkpoint_sha256'),@('report.json','report_sha256'))){if((Get-FileHash "$step/update/$($entry[0])").Hash.ToLowerInvariant() -ne $seal.($entry[1])){throw 'Update completion receipt mismatch'}}
            }
            $current="$step/update/weights.json";$checkpoint="$step/update/checkpoint.pt"
            $records+=@{model=$m.id;epoch=$epoch;episodes=$cfg.episodes;allocated_episodes=($cfg.episodes_per_scene*$cfg.episodes.Count);eligible_transitions=$update.rows;actor_steps=$update.actor_steps;updates_completed=$update.updates_completed;total_actor_steps=$update.total_actor_steps;weights_sha256=$update.weights_sha256;checkpoint=$checkpoint;members=(Get-Content "$step/rollout/report.json" -Raw|ConvertFrom-Json).members}
            $records|ConvertTo-Json -Depth 7|Set-Content "$snapshotRoot/training-updates.json" -Encoding utf8
        }
        foreach($label in @('before','after')){
            $evalModel=if($label -eq 'before'){"$folder/initial.json"}else{$current}
            $evalWeights=Get-Content $evalModel -Raw|ConvertFrom-Json;$evalWeights.deterministic=$true
            $evalWeights|ConvertTo-Json -Depth 100|Set-Content "$folder/eval-$label.json" -Encoding utf8
            @{stage='validation';model=$m.id;label=$label}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
            & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.evaluation_episodes -join ',') --split validation --mode learned --model "$folder/eval-$label.json" --count $cfg.evaluation_count --out "$folder/eval-$label-plan.json" --artifacts "$folder/evaluation-$label"
            if($LASTEXITCODE){throw 'Validation plan failed'}
            & "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan "$folder/eval-$label-plan.json" -Port $Port
            $evaluations+=@{model=$m.id;label=$label;root="$folder/evaluation-$label";weights_sha256=(Get-FileHash "$folder/eval-$label.json").Hash.ToLowerInvariant();source_weights_sha256=(Get-FileHash $evalModel).Hash.ToLowerInvariant()}
        }
    }
    if($cfg.include_rules){& "$PSScriptRoot/run_registered_combat_episodes.ps1" -Plan "$snapshotRoot/preflight-rules.json" -Port $Port;$evaluations+=@{model='rules';label='reference';root="$snapshotRoot/rules"}}
    $board=@()
    foreach($evaluation in $evaluations){
        $runs=Get-Content "$($evaluation.root)/report.json" -Raw|ConvertFrom-Json
        foreach($binding in $runs.records){
            $report=Get-Content "$($binding.artifacts)/report.json" -Raw|ConvertFrom-Json
            if($report.capture_complete){
                $metrics=@{runs=$report.results.Count;wins=@($report.results|Where-Object goal_stop).Count;deaths=@($report.results|Where-Object {$_.first_life.end_reason -eq 'first_observed_death'}).Count;kills=($report.results|ForEach-Object {$_.first_life.damage.by_mod}|Measure-Object monster_kills -Sum).Sum;received_damage=($report.results|ForEach-Object {$_.first_life.damage}|Measure-Object received_health_damage -Sum).Sum}
            }else{$metrics=@{runs=$report.results.Count;wins=@($report.results|Where-Object completed).Count;deaths=($report.results|Measure-Object deaths -Sum).Sum;kills=($report.results|Measure-Object monsters_killed -Sum).Sum;received_damage=($report.results|Measure-Object received_health_damage -Sum).Sum}}
            $board+=@{model=$evaluation.model;label=$evaluation.label;episode=$binding.episode_id;metrics=$metrics;artifacts=$binding.artifacts}
        }
    }
    @{state='complete';comparison_kind=$cfg.comparison_kind;training=$records;evaluations=$evaluations;leaderboard=$board;promotion='not assessed; small validation is diagnostic'}|ConvertTo-Json -Depth 10|Set-Content "$snapshotRoot/report.json" -Encoding utf8
    @{stage='complete'}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
    Write-Output $snapshotRoot
}catch{
    @{stage='failed';reason=$_.Exception.Message}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
    throw
}finally{Pop-Location}
