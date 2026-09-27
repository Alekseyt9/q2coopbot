[CmdletBinding()]
param([int[]]$Timescales=@(2),[int]$Port=30046,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_elevator_teammate_trial.ps1" -SideWalls -Config "$PSScriptRoot/scenarios/base2-elevator-ascent-suite.json" -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
