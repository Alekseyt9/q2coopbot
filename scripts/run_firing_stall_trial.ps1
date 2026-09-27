[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29820,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_parasite_spacing_trial.ps1" -FiringStall -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
