[CmdletBinding()]
param(
 [string]$Model='workspace/artifacts/combat-obstacle-escape-v4-20261006/update/weights.json',
 [string]$Bank='workspace/artifacts/combat-remaining-far-v4-20261006/control-bank.json',
 [string]$Config='scripts/scenarios/combat-ppo-maneuver-v4.json',
 [string]$Reward='scripts/scenarios/combat-reward-maneuver-v4.json',
 [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',
 [string]$OutputRoot='workspace/artifacts/combat-architecture-v1r2-20261006',
 [int]$Seed=26000,[int]$Port=34100
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
 if(Test-Path -LiteralPath $OutputRoot){throw 'Output already exists; preserve evidence and choose a new root'}
 $Model=(Resolve-Path -LiteralPath $Model).Path;$Bank=(Resolve-Path -LiteralPath $Bank).Path
 $Config=(Resolve-Path -LiteralPath $Config).Path;$Reward=(Resolve-Path -LiteralPath $Reward).Path
 if($Seed -lt 0 -or [long]$Seed+303 -gt 2147483647){throw 'Invalid seed range'}
 & $Python scripts/combat_retention_bank.py --validate $Bank --anchor-model $Model --ppo-seed $Seed --eval-seed ($Seed+100) --iterations 2
 if($LASTEXITCODE){throw 'Bank validation failed'}
 New-Item -ItemType Directory -Path $OutputRoot | Out-Null
 $OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
 Copy-Item -LiteralPath $Config -Destination "$OutputRoot/config.json"
 Copy-Item -LiteralPath $Reward -Destination "$OutputRoot/reward.json"
 Copy-Item -LiteralPath $Bank -Destination "$OutputRoot/bank.json"
 Copy-Item -LiteralPath $Model -Destination "$OutputRoot/parent.json"
	New-Item -ItemType Directory -Path "$OutputRoot/parent" | Out-Null
 $Model="$OutputRoot/parent.json";$Bank="$OutputRoot/bank.json";$Config="$OutputRoot/config.json";$Reward="$OutputRoot/reward.json"
 @{seed=$Seed;iterations=2;workers=4;timescale=2;game_frames=300;release_game_frame=100;parent_sha256=(Get-FileHash $Model).Hash;config_sha256=(Get-FileHash $Config).Hash;reward_sha256=(Get-FileHash $Reward).Hash;bank_sha256=(Get-FileHash $Bank).Hash}|ConvertTo-Json|Set-Content -LiteralPath "$OutputRoot/protocol.json" -Encoding utf8NoBOM
 foreach($arm in @('mlp','gru','attention','entity')) {
  New-Item -ItemType Directory -Path "$OutputRoot/$arm" | Out-Null
  Copy-Item -LiteralPath $Config -Destination "$OutputRoot/$arm/config.json"
  if($arm -eq 'mlp') {New-Item -ItemType Directory -Path "$OutputRoot/$arm/initial" | Out-Null;Copy-Item -LiteralPath $Model -Destination "$OutputRoot/$arm/initial/weights.json"}
  else {& $Python scripts/ppo_recurrent.py --init --architecture $arm --model $Model --out "$OutputRoot/$arm/initial";if($LASTEXITCODE){throw "Initialization $arm failed"}}
 }
 $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
 & go build -o workspace/build/q2ppo-data-architecture.exe ./cmd/q2ppo-data
 if($LASTEXITCODE){throw 'Exporter build failed'}
 $exporter=(Resolve-Path workspace/build/q2ppo-data-architecture.exe).Path
 # Both baseline and augmented models start from identical policy outputs.
 # Adam is fresh in every arm, including MLP; no old optimizer is inherited.
 foreach($arm in @('mlp','gru','attention','entity')) {
  $weights="$OutputRoot/$arm/initial/weights.json";$checkpoint=''
  foreach($iteration in 1..2) {
   $folder="$OutputRoot/$arm/iteration-$iteration";New-Item -ItemType Directory -Path $folder | Out-Null
   & scripts/run_learned_combat_baseline.ps1 -Workers 4 -EpisodesPerWorker 1 -Timescale 2 -GameFrames 300 -ReleaseGameFrame 100 -Mixed -Loadout blaster -Skill 1 -CombatMode learned -ProviderFile $weights -Synchronous -RewardConfig $Reward -Seed ($Seed+4*($iteration-1)) -Port $Port -OutputRoot "$folder/batch"
   & $exporter --batch "$folder/batch" --model $weights --out "$folder/rollout"
   if($LASTEXITCODE){throw "$arm iteration$iteration native export failed"}
   $trainer=if($arm -eq 'mlp'){'scripts/ppo_combat.py'}else{'scripts/ppo_recurrent.py'}
   $updateArgs=@($trainer,'--model',$weights,'--data',"$folder/rollout",'--config',$Config,'--out',"$folder/update",'--anchor-model',$Model,'--retention-bank',$Bank)
   if($arm -eq 'mlp'){$updateArgs+=@('--retention-mode','constant','--retention-weight','1','--retention-updates','2','--bank-weight','1')}
   if($checkpoint){$updateArgs+=@('--resume',$checkpoint)}
   & $Python @updateArgs
   if($LASTEXITCODE){throw "$arm iteration$iteration CUDA update failed"}
   $weights="$folder/update/weights.json";$checkpoint="$folder/update/checkpoint.pt"
  }
 }
 foreach($fixture in @('mixed','solo','hard-solo')) {
  $evalSeed=switch($fixture){'mixed'{$Seed+100};'solo'{$Seed+200};'hard-solo'{$Seed+300}}
  foreach($arm in @('parent','mlp','gru','attention','entity')) {
   $input=if($arm -eq 'parent'){$Model}else{"$OutputRoot/$arm/iteration-2/update/weights.json"}
   $output="$OutputRoot/$arm/eval-$fixture.json";$m=Get-Content -LiteralPath $input -Raw|ConvertFrom-Json;$m.deterministic=$true
   $m|ConvertTo-Json -Depth 16|Set-Content -LiteralPath $output -Encoding utf8NoBOM
   & scripts/run_learned_combat_baseline.ps1 -Workers 4 -EpisodesPerWorker 1 -Timescale 2 -GameFrames 300 -ReleaseGameFrame 100 -Mixed:($fixture -eq 'mixed') -Loadout blaster -Skill $(if($fixture -eq 'hard-solo'){3}else{1}) -CombatMode learned -ProviderFile $output -Synchronous -RewardConfig $Reward -Seed $evalSeed -Port $Port -OutputRoot "$OutputRoot/$arm/evaluation-$fixture"
  }
 }
 & $Python scripts/audit_combat_architecture.py --root $OutputRoot
 if($LASTEXITCODE){throw 'Final architecture audit failed'}
}finally{Pop-Location}
