[CmdletBinding()]
param([Parameter(Mandatory)][string]$CaptureRoot,[Parameter(Mandatory)][string]$ProcessingRoot,[int]$Port=34600)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=(Resolve-Path -LiteralPath $CaptureRoot).Path
$previous=Join-Path $repo 'workspace/artifacts/aproc-v1-20261007'
$plans=@(Get-Content -LiteralPath "$root/plans.json" -Raw|ConvertFrom-Json)
& "$PSScriptRoot/run_registered_combat_episode_pool.ps1" -Plans $plans -MaxInstances 16 -Port $Port -OutputRoot "$root/pool"
& 'F:/src/strat/.venv-gpu/Scripts/python.exe' "$PSScriptRoot/process_combat_architecture_pool.py" --capture-root $root --out $ProcessingRoot --config "$previous/config.json" --exporter "$previous/q2ppo-data.exe" --anchor "$previous/anchor.json" --bank "$previous/bank.json" --retry-port ($Port+20) --export-workers 4
if($LASTEXITCODE){throw 'Own-policy CUDA processing failed'}
