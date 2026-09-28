param([int[]]$Timescales=@(2),[int]$Port=30410,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_death_point_return.ps1"
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2 | Out-Null
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/death-point-return-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
New-Item -ItemType Directory $OutputRoot -Force|Out-Null
$cfg=Get-Content "$PSScriptRoot/scenarios/death-point-return-suite.json" -Raw|ConvertFrom-Json
$cfg.sessions=@((Join-Path $PSScriptRoot 'scenarios/death-point-return-session.json'))
$cfg.runtime_root=Join-Path $repo 'workspace/runtime/q2go-elevator-cycle-base2'
$base=(Resolve-Path $OutputRoot).Path
$cfg.sessions=@($cfg.sessions|ForEach-Object {[IO.Path]::GetRelativePath($base,$_ )})
$cfg.runtime_root=[IO.Path]::GetRelativePath($base,$cfg.runtime_root)
$cfg.timescales=$Timescales;$cfg.base_port=$Port
$path=Join-Path $OutputRoot 'suite.json';$cfg|ConvertTo-Json -Depth 5|Set-Content $path
$messages=@(& "$PSScriptRoot/run_scenario_suite.ps1" -Config $path)
$line=@($messages | Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'})
if($line.Count -ne 1){throw 'Missing suite result'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$root=$Matches[1]
$suite=Get-Content (Join-Path $root 'suite-summary.json') -Raw|ConvertFrom-Json
$details=@(foreach($run in $suite.results){
    if(!$run.accepted -or !$suite.provenance_valid){throw 'Rejected session'}
    [pscustomobject]@{timescale=$run.timescale;cases=@(Assert-DeathPointReturn @(Get-Content (Join-Path $run.directory 'observer.jsonl')|ConvertFrom-Json))}
})
@(foreach($scale in $Timescales){[pscustomobject]@{timescale=$scale;accepted=$true;details=@($details|Where-Object timescale -eq $scale);suite=$root}})|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')
$details
