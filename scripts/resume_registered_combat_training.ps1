# Resume a fully captured single-epoch experiment without replaying its battles.
[CmdletBinding()]
param([Parameter(Mandatory)][string]$OutputRoot,[string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',[int]$Port=34400)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$snapshotRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$boundary=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts')).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
if(!$snapshotRoot.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Resume requires an owned artifact experiment'}
$cfg=Get-Content "$snapshotRoot/config.json" -Raw|ConvertFrom-Json
$protocol=Get-Content "$snapshotRoot/protocol.json" -Raw|ConvertFrom-Json
if($cfg.epochs -ne 1 -or $cfg.include_rules){throw 'Capture resume currently supports one epoch without rules'}
if(Test-Path "$snapshotRoot/report.json"){throw 'Completed experiment cannot be resumed'}
$poolInstances=[int]$protocol.pool_instances
if($poolInstances -ne 16){throw 'Current working pool capacity is 16'}
$evaluationOffset=[int]$protocol.evaluation_seed_offset
$registry=Join-Path $repo $cfg.registry
$trainingConfig="$snapshotRoot/training-config.json";$anchor="$snapshotRoot/anchor.json";$bank="$snapshotRoot/bank.json"
if((Get-FileHash $registry).Hash.ToLowerInvariant() -ne $protocol.registry_sha256 -or (Get-FileHash $trainingConfig).Hash.ToLowerInvariant() -ne $protocol.training_config_sha256){throw 'Frozen registry/objective mismatch'}
foreach($module in $protocol.trainer_modules){if((Get-FileHash "$snapshotRoot/python-sources/$($module.file)").Hash.ToLowerInvariant() -ne $module.sha256){throw 'Frozen trainer module mismatch'}}
$records=@();$evaluations=@();$states=@();$epoch=1
foreach($m in $protocol.models){
    $folder="$snapshotRoot/$($m.id)";$current="$folder/initial.json";$checkpoint="$folder/initial-checkpoint.pt";$step="$folder/epoch-1"
    if((Get-FileHash $current).Hash.ToLowerInvariant() -ne $m.initial_sha256){throw 'Frozen initial weights mismatch'}
    if($m.checkpoint_sha256){if((Get-FileHash $checkpoint).Hash.ToLowerInvariant() -ne $m.checkpoint_sha256){throw 'Frozen initial checkpoint mismatch'}}else{$checkpoint=''}
    if(Test-Path "$step/update"){throw 'Existing update requires separate completion audit; refusing a second update'}
    $capture=Get-Content "$step/capture/report.json" -Raw|ConvertFrom-Json
    if($capture.state -ne 'complete' -or $capture.records.Count -ne $cfg.episodes.Count){throw 'Full capture required'}
    foreach($episode in $cfg.episodes){
        $binding=@($capture.records|Where-Object {$_.episode_id -eq $episode -and $_.mode -eq 'learned'})
        if($binding.Count -ne 1 -or !$binding[0].capture_verified -or $binding[0].seeds.Count -ne $cfg.episodes_per_scene -or (Get-FileHash "$($binding[0].artifacts)/report.json").Hash.ToLowerInvariant() -ne $binding[0].report_sha256){throw 'Capture binding mismatch'}
    }
    $states+=@{spec=$m;folder=$folder;current=$current;checkpoint=$checkpoint;capture=$capture}
}
Push-Location $repo
try{
    & $Python -c 'import sys;sys.path.insert(0,sys.argv[1]);from train_combat_bc import training_devices;assert training_devices()==["cuda"]' "$snapshotRoot/python-sources"
    if($LASTEXITCODE){throw 'CUDA-only training unavailable'}
    foreach($state in $states){
        $m=$state.spec;$folder=$state.folder;$current=$state.current;$checkpoint=$state.checkpoint;$step="$folder/epoch-1"
        $resumeRoot="$step/resume-$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
        New-Item -ItemType Directory -Path $resumeRoot|Out-Null
        @{stage='verified-reexport';model=$m.id;resume_root=$resumeRoot}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
        $exports=@()
        foreach($episode in $cfg.episodes){
            $binding=@($state.capture.records|Where-Object episode_id -eq $episode)[0]
            $export="$resumeRoot/rollout-$episode"
            & "$snapshotRoot/q2ppo-data.exe" --batch $binding.artifacts --model $current --out $export
            if($LASTEXITCODE){throw 'Verified on-policy re-export failed'}
            $exports+=$export
        }
        & "$snapshotRoot/q2ppo-data.exe" --merge ($exports -join ',') --out "$resumeRoot/rollout"
        if($LASTEXITCODE){throw 'Verified joint merge failed'}
        $trainer=if($m.architecture -eq 'mlp'){'ppo_combat.py'}else{'ppo_recurrent.py'}
        $argsList=@("$snapshotRoot/python-sources/$trainer",'--model',$current,'--data',"$resumeRoot/rollout",'--config',$trainingConfig,'--out',"$step/update",'--retention-weight','0','--bank-weight','0')
        if($m.architecture -ne 'mlp'){$argsList+=@('--anchor-model',$anchor,'--retention-bank',$bank)}
        if($checkpoint){$argsList+=@('--resume',$checkpoint)}
        @{stage='cuda-update';model=$m.id;epoch=1}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
        & $Python @argsList > "$step/update.stdout" 2> "$step/update.stderr"
        if($LASTEXITCODE){throw 'CUDA update failed'}
        $update=Get-Content "$step/update/report.json" -Raw|ConvertFrom-Json
        $seal=Get-Content "$step/update/complete.json" -Raw|ConvertFrom-Json
        if($update.device -ne 'cuda' -or $update.weights_sha256 -ne $seal.weights_sha256){throw 'Unverified CUDA update'}
        foreach($entry in @(@('weights.json','weights_sha256'),@('checkpoint.pt','checkpoint_sha256'),@('report.json','report_sha256'))){if((Get-FileHash "$step/update/$($entry[0])").Hash.ToLowerInvariant() -ne $seal.($entry[1])){throw 'Completion receipt mismatch'}}
        $state.current="$step/update/weights.json";$state.checkpoint="$step/update/checkpoint.pt"
        $records+=@{model=$m.id;epoch=1;episodes=$cfg.episodes;allocated_episodes=($cfg.episodes_per_scene*$cfg.episodes.Count);eligible_transitions=$update.rows;actor_steps=$update.actor_steps;updates_completed=$update.updates_completed;total_actor_steps=$update.total_actor_steps;weights_sha256=$update.weights_sha256;checkpoint=$state.checkpoint;members=(Get-Content "$resumeRoot/rollout/report.json" -Raw|ConvertFrom-Json).members;resume_root=$resumeRoot}
        $records|ConvertTo-Json -Depth 7|Set-Content "$snapshotRoot/training-updates.json"
    }
    $evaluationPlans=@()
    foreach($state in $states){
        $m=$state.spec;$folder=$state.folder;$current=$state.current
        foreach($label in @('before','after')){
            $evalModel=if($label -eq 'before'){"$folder/initial.json"}else{$current}
            $evalWeights=Get-Content $evalModel -Raw|ConvertFrom-Json;$evalWeights.deterministic=$true
            $evalWeights|ConvertTo-Json -Depth 100|Set-Content "$folder/eval-$label.json" -Encoding utf8
            @{stage='validation';model=$m.id;label=$label}|ConvertTo-Json|Set-Content "$snapshotRoot/progress.json"
            & "$snapshotRoot/q2episode.exe" --registry $registry --root $repo --episodes ($cfg.evaluation_episodes -join ',') --split validation --mode learned --model "$folder/eval-$label.json" --count $cfg.evaluation_count --seed-offset $evaluationOffset --out "$folder/eval-$label-plan.json" --artifacts "$folder/evaluation-$label"
            if($LASTEXITCODE){throw 'Validation plan failed'}
            $evaluationPlans+=@("$folder/eval-$label-plan.json")
            $evaluations+=@{model=$m.id;label=$label;root="$folder/evaluation-$label";weights_sha256=(Get-FileHash "$folder/eval-$label.json").Hash.ToLowerInvariant();source_weights_sha256=(Get-FileHash $evalModel).Hash.ToLowerInvariant()}
        }
    }
    if($poolInstances -eq 4){foreach($plan in $evaluationPlans){Invoke-RegisteredCapture $plan ''}}
    else{& "$PSScriptRoot/run_registered_combat_pool.ps1" -Plans $evaluationPlans -MaxInstances $poolInstances -Port $Port -OutputRoot "$snapshotRoot/evaluation-pool"}
    if($cfg.include_rules){Invoke-RegisteredCapture "$snapshotRoot/preflight-rules.json" "$snapshotRoot/rules-pool";$evaluations+=@{model='rules';label='reference';root="$snapshotRoot/rules"}}
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

