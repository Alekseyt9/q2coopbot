[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29780,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_moving_teammate_trial.ps1" -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot -ProjectileCrossing
