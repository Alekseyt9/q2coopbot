[CmdletBinding()]
param([Parameter(Mandatory)][string]$EvaluationRoot,[int]$Port=34400)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $EvaluationRoot).Path
$plans=@(Get-Content -LiteralPath "$root/plans.json" -Raw|ConvertFrom-Json)
@{stage='capturing';slots=16;timescale=2}|ConvertTo-Json|Set-Content -LiteralPath "$root/progress.json"
& "$PSScriptRoot/run_registered_combat_episode_pool.ps1" -Plans $plans -MaxInstances 16 -Port $Port -OutputRoot "$root/pool"
& 'F:/src/strat/.venv-gpu/Scripts/python.exe' "$PSScriptRoot/report_combat_architecture_evaluation.py" --root $root
if($LASTEXITCODE){throw 'Paired quality report failed'}
