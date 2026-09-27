[CmdletBinding()]
param([int[]]$Timescales=@(2),[int]$Port=30126,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_elevator_teammate_trial.ps1" -Config "$PSScriptRoot/scenarios/base2-elevator-late-controller-suite.json" -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
