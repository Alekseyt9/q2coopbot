[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29540,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
& "$PSScriptRoot/run_projectile_comparison.ps1" -Range far -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot
