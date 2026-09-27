[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29860,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_wall_coast_brake_trial.ps1" -Edge -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
