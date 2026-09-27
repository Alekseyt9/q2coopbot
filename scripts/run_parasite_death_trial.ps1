param([int[]]$Timescales=@(2),[int]$Port=30230,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/parasite-death-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
. "$PSScriptRoot/check_parasite_death.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
& "$PSScriptRoot/run_combat_run_in_trial.ps1" -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
$report=@(foreach($run in @(Get-Content (Join-Path $OutputRoot 'summary.json') -Raw | ConvertFrom-Json)) {
    $detail=Assert-ParasiteDeath @(Get-Content $run.trace_jsonl | ConvertFrom-Json) @(Read-DamageEvents $run.combat_log)
    [pscustomobject]@{accepted=$detail.accepted;timescale=$run.timescale;detail=$detail;trace=$run.trace_jsonl}
})
$report | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputRoot 'report.json')
$report
