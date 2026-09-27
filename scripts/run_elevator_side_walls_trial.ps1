[CmdletBinding()]
param([int[]]$Timescales=@(2),[int]$Port=30026,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_elevator_teammate_trial.ps1" -SideWalls -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
