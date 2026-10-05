[CmdletBinding()]
param([string]$Spec='',[string]$OutputRoot='',[ValidateRange(20,500)][int]$GameFrames=300,[ValidateRange(1024,65530)][int]$Port=33028)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$Spec){$Spec=Join-Path $PSScriptRoot 'scenarios/combat-dataset-v1.json'}
$plan=Get-Content -LiteralPath $Spec -Raw|ConvertFrom-Json
if($plan.condition.map -ne 'base1' -or $plan.condition.loadout -notin 'blaster','shotgun' -or !$plan.condition.synchronous){throw 'Current demonstration runner requires a frozen synchronous base1 Blaster or Shotgun condition'}
$seeds=@($plan.episodes.seed|Sort-Object)
if($seeds.Count -lt 4 -or $seeds.Count%4 -ne 0 -or @($seeds|Select-Object -Unique).Count -ne $seeds.Count -or $seeds[-1]-$seeds[0]+1 -ne $seeds.Count){throw 'Four workers require unique contiguous planned seeds in multiples of four'}
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/combat-demonstrations-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output directory required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$frozen=Join-Path $OutputRoot 'frozen-spec.json'
Copy-Item -LiteralPath $Spec -Destination $frozen
$specHash=(Get-FileHash -LiteralPath $frozen).Hash
$builder=Join-Path $OutputRoot 'q2combat-dataset.exe'
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache';$env:GOTOOLCHAIN='auto'
Push-Location $repo;try{go build -o $builder ./cmd/q2combat-dataset;if($LASTEXITCODE){throw 'Dataset builder failed'}}finally{Pop-Location}
$baseline=Join-Path $OutputRoot 'baseline'
& "$PSScriptRoot/run_learned_combat_baseline.ps1" -Workers 4 -EpisodesPerWorker ($seeds.Count/4) -Timescale 2 -GameFrames $GameFrames -Loadout $plan.condition.loadout -CombatMode rules -Synchronous -TeacherVertical:([bool]$plan.condition.teacher_vertical) -Mixed:([bool]$plan.condition.mixed) -HealthKit:([bool]$plan.condition.health_kit) -RewardConfig (Join-Path $PSScriptRoot 'scenarios/combat-reward-v1.json') -Seed $seeds[0] -Port $Port -OutputRoot $baseline
if($specHash -ne (Get-FileHash -LiteralPath $frozen).Hash){throw 'Frozen split specification changed'}
& $builder --spec $frozen --batch $baseline --out (Join-Path $OutputRoot 'dataset')
if($LASTEXITCODE){throw 'Dataset preparation rejected; inspect preserved diagnostics'}
"Demonstrations: $OutputRoot"
