[CmdletBinding()]
param(
 [string]$Model='workspace/artifacts/combat-architecture-v2-20261006/attention/iteration-10/update/weights.json',
 [string]$Anchor='workspace/artifacts/combat-architecture-v2-20261006/parent.json',
 [string]$Bank='workspace/artifacts/combat-architecture-v2-20261006/bank.json',
 [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',
 [string]$OutputRoot='workspace/artifacts/combat-mobile-machinegun-v1r2-20261006',
 [string]$Config='scripts/scenarios/combat-ppo-maneuver-v4.json',
 [string]$Reward='scripts/scenarios/combat-reward-maneuver-v4.json',
 [string]$InitialCheckpoint='',
 [ValidateRange(1,20)][int]$EpisodesPerWorker=1,
 [ValidateSet('uniform','mixed-solo-mixed')][string]$EpisodePattern='uniform',
 [ValidateRange(1,20)][int]$EvalEpisodesPerWorker=1,
 [ValidateRange(1,20)][int]$Iterations=4,
 [int]$Seed=30000,[int]$EvalSeed=30400,[int]$Port=34400
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
 if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output root required'}
 if($EpisodePattern -ne 'uniform' -and $EpisodesPerWorker -ne 3){throw 'Mixed/Solo pattern requires exactly three episodes per worker'}
 $batchWidth=4*$EpisodesPerWorker
 if($EvalSeed -le $Seed+$batchWidth*$Iterations-1 -and $EvalSeed+4*$EvalEpisodesPerWorker-1 -ge $Seed){throw 'Evaluation seeds overlap training'}
 $Model=(Resolve-Path $Model).Path;$Anchor=(Resolve-Path $Anchor).Path;$Bank=(Resolve-Path $Bank).Path
 if($InitialCheckpoint){
  $InitialCheckpoint=(Resolve-Path $InitialCheckpoint).Path
  & $Python -c 'import sys,hashlib,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=True); assert c["weights_sha256"]==hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(); assert c["retention_weights"]==[0.,0.]' $InitialCheckpoint $Model
  if($LASTEXITCODE){throw 'Initial checkpoint does not match model or mobile retention settings'}
 }
 & $Python scripts/combat_retention_bank.py --validate $Bank --anchor-model $Anchor --ppo-seed $Seed --eval-seed $EvalSeed --iterations $Iterations --episodes-per-worker $EpisodesPerWorker --eval-episodes-per-worker $EvalEpisodesPerWorker
 if($LASTEXITCODE){throw 'Reference provenance validation failed'}
 New-Item -ItemType Directory -Path $OutputRoot | Out-Null
 $OutputRoot=(Resolve-Path $OutputRoot).Path
 Copy-Item -LiteralPath $Model -Destination "$OutputRoot/initial.json"
 Copy-Item -LiteralPath $Anchor -Destination "$OutputRoot/anchor.json"
 Copy-Item -LiteralPath $Bank -Destination "$OutputRoot/bank.json"
 Copy-Item -LiteralPath $Config -Destination "$OutputRoot/config.json"
 Copy-Item -LiteralPath $Reward -Destination "$OutputRoot/reward.json"
 New-Item -ItemType Directory -Path "$OutputRoot/python-sources" | Out-Null
 foreach($module in @('ppo_combat.py','ppo_recurrent.py','combat_attention.py','combat_retention_bank.py','train_combat_bc.py')){Copy-Item -LiteralPath "scripts/$module" -Destination "$OutputRoot/python-sources/$module"}
 $weights="$OutputRoot/initial.json";$checkpoint=''
 if($InitialCheckpoint){Copy-Item -LiteralPath $InitialCheckpoint -Destination "$OutputRoot/initial-checkpoint.pt";$checkpoint="$OutputRoot/initial-checkpoint.pt"}
 $initial=Get-Content $weights -Raw|ConvertFrom-Json
 if(!$initial.attention -or $initial.deterministic){throw 'Requires stochastic Temporal attention weights'}
 @{version='mobile_machinegun_v1';initial_sha256=(Get-FileHash $weights).Hash;iterations=$Iterations;workers=4;timescale=2;training_seeds=@($Seed..($Seed+4*$Iterations-1));evaluation_seeds=@($EvalSeed..($EvalSeed+3));loadout='machinegun';bullets=100;skill=1;monster_spawn_height=24.125;monster_no_infighting=$true;retention_weight=0;bank_weight=0;optimizer='fresh Adam; resume only within corrected cycle';scope='Historical Temporal10 is a warm start; old stationary captures are not training losses or mobile evaluation evidence'}|ConvertTo-Json -Depth 5|Set-Content "$OutputRoot/protocol.json"
 $protocol=Get-Content "$OutputRoot/protocol.json" -Raw|ConvertFrom-Json
 $protocol.training_seeds=@($Seed..($Seed+$batchWidth*$Iterations-1))
 $protocol.evaluation_seeds=@($EvalSeed..($EvalSeed+4*$EvalEpisodesPerWorker-1))
 $protocol|Add-Member -NotePropertyName stop_on_goal -NotePropertyValue $true
 $protocol|Add-Member -NotePropertyName episodes_per_worker -NotePropertyValue $EpisodesPerWorker
 $protocol|Add-Member -NotePropertyName episode_pattern -NotePropertyValue $EpisodePattern
 $protocol|Add-Member -NotePropertyName eval_episodes_per_worker -NotePropertyValue $EvalEpisodesPerWorker
 $protocol|ConvertTo-Json -Depth 5|Set-Content "$OutputRoot/protocol.json"
 if($InitialCheckpoint){
  $protocol=Get-Content "$OutputRoot/protocol.json" -Raw|ConvertFrom-Json
  $protocol.optimizer='Resume Adam from verified matching mobile checkpoint'
  $protocol|Add-Member -NotePropertyName initial_checkpoint_sha256 -NotePropertyValue (Get-FileHash "$OutputRoot/initial-checkpoint.pt").Hash
  $protocol|ConvertTo-Json -Depth 5|Set-Content "$OutputRoot/protocol.json"
 }
 $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
 & go build -o "$OutputRoot/q2ppo-data.exe" ./cmd/q2ppo-data
 if($LASTEXITCODE){throw 'Native exporter build failed'}
 $acceptedFingerprint='';$acceptedNativeFingerprint=''
 function Check-MobileBatch([string]$Batch) {
  $report=Get-Content "$Batch/report.json" -Raw|ConvertFrom-Json
  $manifest=Get-Content "$Batch/manifest.json" -Raw|ConvertFrom-Json
  if(!$report.capture_complete -or !$report.provenance_valid -or !$manifest.monster_no_infighting){throw 'Invalid mobile capture provenance'}
  if($script:acceptedFingerprint -and ($manifest.source_fingerprint -ne $script:acceptedFingerprint -or $manifest.native_source_fingerprint -ne $script:acceptedNativeFingerprint)){throw 'Source changed between corrected batches'}
  $script:acceptedFingerprint=$manifest.source_fingerprint;$script:acceptedNativeFingerprint=$manifest.native_source_fingerprint
  $proof=@(foreach($episode in $report.results){
   $run=Join-Path $Batch ("worker-$($episode.worker)-episode-$($episode.episode)")
   if(!(Test-Path "$run/bot.jsonl")){throw "Missing run trace $run"}
   $rows=@(Get-Content "$run/bot.jsonl"|ForEach-Object{$_|ConvertFrom-Json}|Where-Object frame -ge 100)
   $log=Get-Content "$run/server.log"
   if(!($log|Select-String '^g_test_monster_no_infighting ready version=1 enabled=1')){throw 'Native no-infighting not acknowledged'}
   # Stock crusher damage may retain an enemy as attacker on an already dead
   # monster. Corpses have lost SVF_MONSTER; their damage is not infighting.
   if($log|Select-String 'g_test_damage version=.*health_before=[1-9][0-9]* .*target_class=monster_\w+ attacker_class=monster_'){throw 'Native live monster friendly damage occurred'}
   $expectedClasses=@('monster_parasite')
   if($episode.fixture_mixed){$expectedClasses+='monster_gunner'}
   $positions=@{};foreach($class in $expectedClasses){
    $points=@($rows|ForEach-Object{$_.enemies|Where-Object class -eq $class}|ForEach-Object{"$($_.origin[0]),$($_.origin[1])"}|Select-Object -Unique)
    $positions[$class]=$points.Count
    if($points.Count -lt 2){throw "Monster mobility not verified for $class seed$($episode.seed)"}
   }
   @{seed=$episode.seed;horizontal_positions=$positions;blocked_monster_contacts=@($log|Select-String '^g_test_monster_no_infighting blocked').Count;no_monster_damage=$true}
  })
  $proof|ConvertTo-Json -Depth 5|Set-Content "$Batch/mobile-proof.json"
 }
 foreach($iteration in 1..$Iterations){
  $folder="$OutputRoot/iteration-$iteration";New-Item -ItemType Directory -Path $folder|Out-Null
  @{stage='capture';iteration=$iteration;iterations=$Iterations}|ConvertTo-Json|Set-Content "$OutputRoot/progress.json"
  & scripts/run_learned_combat_baseline.ps1 -Workers 4 -EpisodesPerWorker $EpisodesPerWorker -EpisodePattern $EpisodePattern -Timescale 2 -GameFrames 300 -ReleaseGameFrame 100 -Mixed -Loadout machinegun -Skill 1 -CombatMode learned -ProviderFile $weights -Synchronous -StopOnGoal -RewardConfig "$OutputRoot/reward.json" -Seed ($Seed+$batchWidth*($iteration-1)) -Port $Port -OutputRoot "$folder/batch"
  Check-MobileBatch "$folder/batch"
  & "$OutputRoot/q2ppo-data.exe" --batch "$folder/batch" --model $weights --out "$folder/rollout"
  if($LASTEXITCODE){throw 'Native rollout export failed'}
  @{stage='cuda-update';iteration=$iteration;iterations=$Iterations}|ConvertTo-Json|Set-Content "$OutputRoot/progress.json"
  $argsList=@("$OutputRoot/python-sources/ppo_recurrent.py",'--model',$weights,'--data',"$folder/rollout",'--config',"$OutputRoot/config.json",'--out',"$folder/update",'--anchor-model',"$OutputRoot/anchor.json",'--retention-bank',"$OutputRoot/bank.json",'--retention-weight','0','--bank-weight','0')
  if($checkpoint){$argsList+=@('--resume',$checkpoint)}
  & $Python @argsList
  if($LASTEXITCODE){throw 'CUDA update failed'}
  $seal=Get-Content "$folder/update/complete.json" -Raw|ConvertFrom-Json
  if($seal.version -ne 'combat_update_complete_v1'){throw 'Missing completed CUDA update receipt'}
  foreach($entry in @(@('weights.json','weights_sha256'),@('checkpoint.pt','checkpoint_sha256'),@('report.json','report_sha256'))){
   if((Get-FileHash "$folder/update/$($entry[0])").Hash.ToLowerInvariant() -ne $seal.($entry[1])){throw "CUDA output receipt mismatch: $($entry[0])"}
  }
  $weights="$folder/update/weights.json";$checkpoint="$folder/update/checkpoint.pt"
 }
 foreach($label in @('before','after')){
  $modelPath=if($label -eq 'before'){"$OutputRoot/initial.json"}else{$weights}
  $eval=Get-Content $modelPath -Raw|ConvertFrom-Json;$eval.deterministic=$true
  $eval|ConvertTo-Json -Depth 16|Set-Content "$OutputRoot/eval-$label.json" -Encoding utf8NoBOM
  @{stage='evaluation';label=$label;iterations=$Iterations}|ConvertTo-Json|Set-Content "$OutputRoot/progress.json"
  & scripts/run_learned_combat_baseline.ps1 -Workers 4 -EpisodesPerWorker $EvalEpisodesPerWorker -Timescale 2 -GameFrames 300 -ReleaseGameFrame 100 -Mixed -Loadout machinegun -Skill 1 -CombatMode learned -ProviderFile "$OutputRoot/eval-$label.json" -Synchronous -StopOnGoal -RewardConfig "$OutputRoot/reward.json" -Seed $EvalSeed -Port $Port -OutputRoot "$OutputRoot/evaluation-$label"
  Check-MobileBatch "$OutputRoot/evaluation-$label"
 }
 @{stage='complete';iterations=$Iterations;final_weights=$weights;source_fingerprint=$script:acceptedFingerprint;native_source_fingerprint=$script:acceptedNativeFingerprint}|ConvertTo-Json|Set-Content "$OutputRoot/progress.json"
} catch {
 if($OutputRoot -and (Test-Path $OutputRoot)){@{stage='failed';reason=$_.Exception.Message}|ConvertTo-Json|Set-Content "$OutputRoot/progress.json"}
 throw
} finally {Pop-Location}
