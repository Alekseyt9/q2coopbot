[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputRoot,
    [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe',
    [ValidateRange(1,2000)][int]$Epochs=200,
    [ValidateRange(1,2000000000)][int]$Seed=25111500,
    [ValidateRange(1024,65480)][int]$Port=35410
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh pilot output directory required'}
New-Item -ItemType Directory -Path $root | Out-Null
Push-Location $repo
try {
    $runner=Join-Path $PSScriptRoot 'run_learned_combat_baseline.ps1'
    $common=@{Workers=4;Synchronous=$true;Loadout='machinegun';ReleaseGameFrame=100;GameFrames=220}
    & $runner @common -EpisodesPerWorker 2 -CombatMode rules -Seed $Seed -Port $Port -OutputRoot "$root/teacher-train"
    & $runner @common -EpisodesPerWorker 1 -CombatMode rules -Seed ($Seed+100) -Port ($Port+10) -OutputRoot "$root/teacher-validation"
    & go run ./cmd/q2hjepa-data --root "$root/teacher-train" --out "$root/train.jsonl"
    if($LASTEXITCODE){throw 'Train export failed'}
    & go run ./cmd/q2hjepa-data --root "$root/teacher-validation" --out "$root/validation.jsonl"
    if($LASTEXITCODE){throw 'Validation export failed'}
    & $Python "$PSScriptRoot/train_combat_hjepa.py" --train "$root/train.jsonl" --validation "$root/validation.jsonl" --out "$root/model" --epochs $Epochs --seed $Seed | Tee-Object -FilePath "$root/training.log"
    if($LASTEXITCODE){throw 'CUDA training failed'}
    & go run ./cmd/q2hjepa-check --model "$root/model/weights.json" --parity "$root/model/parity.json" | Tee-Object -FilePath "$root/parity-report.json"
    if($LASTEXITCODE){throw 'Go/CUDA parity failed'}
    $common.GameFrames=300
    $common.EpisodesPerWorker=1
    $common.StopOnGoal=$true
    $common.StopOnFirstDeath=$true
    $common.RewardConfig=Join-Path $PSScriptRoot 'scenarios/combat-reward-native-goal-v1.json'
    & $runner @common -CombatMode learned -ProviderFile "$root/model/weights.json" -Seed ($Seed+200) -Port ($Port+20) -OutputRoot "$root/eval-hjepa"
    & $runner @common -CombatMode rules -Seed ($Seed+200) -Port ($Port+30) -OutputRoot "$root/eval-rules"
    & $Python "$PSScriptRoot/report_combat_hjepa.py" --root $root
    if($LASTEXITCODE){throw 'Paired summary failed'}
} finally { Pop-Location }
