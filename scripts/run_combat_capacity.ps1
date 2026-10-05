[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Checkpoint,
    [Parameter(Mandatory)][string]$ProbeRollout,
    [Parameter(Mandatory)][string]$OutputRoot,
    [ValidateRange(1,20)][int]$Iterations=4,
    [int]$Seed=14700,[int]$EvalSeed=14800,
    [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe'
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$Model=(Resolve-Path -LiteralPath $Model).Path
$Checkpoint=(Resolve-Path -LiteralPath $Checkpoint).Path
$ProbeRollout=(Resolve-Path -LiteralPath $ProbeRollout).Path
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$config=Join-Path $PSScriptRoot 'scenarios/combat-ppo-kill-v2.json'
$reward=Join-Path $PSScriptRoot 'scenarios/combat-reward-v2.json'
$fork=Join-Path $PSScriptRoot 'fork_combat_capacity.py'
$cycle=Join-Path $PSScriptRoot 'run_combat_ppo.ps1'
$pins=@{}
foreach($path in @($Model,$Checkpoint,$ProbeRollout,$config,$reward,$fork,$cycle)){$pins[$path]=(Get-FileHash -LiteralPath $path).Hash}
$protocol=@{version='combat_capacity_comparison_v1';widths=@(64,128,256);iterations=$Iterations;workers=4;timescale=2;game_frames=300;training_monster_health=20;evaluation_monster_health=175;training_seed=$Seed;evaluation_seed=$EvalSeed;source_sha256=$pins;checkpoint_selection='fixed final update, no best-of evaluation';scope='Paired episode seeds across widths; independent seed per worker. Same game-frame and update budget; usable rollout counts can differ. Both Adam states reset in all arms; RNG/objective/log_std retained.'}
$protocol|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $OutputRoot 'protocol.json') -Encoding utf8NoBOM
$summaries=@()
Push-Location $repo
try{
    foreach($width in @(64,128,256)){
        foreach($path in $pins.Keys){if((Get-FileHash -LiteralPath $path).Hash -ne $pins[$path]){throw "Pinned input changed: $path"}}
        $arm=Join-Path $OutputRoot "width-$width";New-Item -ItemType Directory -Path $arm|Out-Null
        $init=Join-Path $arm 'initial'
        & $Python $fork --model $Model --checkpoint $Checkpoint --probe-rollout $ProbeRollout --width $width --out $init
        if($LASTEXITCODE){throw "Capacity fork failed: $width"}
        & $cycle -Model (Join-Path $init 'weights.json') -Checkpoint (Join-Path $init 'checkpoint.pt') -Iterations $Iterations -Seed $Seed -EvalSeed $EvalSeed -TrainingMonsterHealth 20 -GameFrames 300 -Config $config -RewardConfig $reward -Python $Python -OutputRoot (Join-Path $arm 'cycle')
        $r=Get-Content -LiteralPath (Join-Path $arm 'cycle/report.json') -Raw|ConvertFrom-Json
        $f=Get-Content -LiteralPath (Join-Path $init 'report.json') -Raw|ConvertFrom-Json
        $summaries+=@{width=$width;total_parameters=$f.total_parameters;initial_parity=$f.max_output_error;cycle_report=(Join-Path $arm 'cycle/report.json');rows=[int](($r.steps.report|Measure-Object rows -Sum).Sum);evaluation=$r.evaluation;fixture_promotion_eligible=$r.fixture_promotion_eligible}
        $summaries|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $OutputRoot 'progress.json') -Encoding utf8NoBOM
    }
    @{version='combat_capacity_comparison_v1';arms=$summaries;scope='Small fixed-budget pilot on one fixture; no automatic live promotion, significance or generalization claim'}|ConvertTo-Json -Depth 14|Set-Content -LiteralPath (Join-Path $OutputRoot 'report.json') -Encoding utf8NoBOM
}finally{Pop-Location}
