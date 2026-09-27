[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29580,[string]$OutputRoot='')
& "$PSScriptRoot/run_projectile_comparison.ps1" -Timescales $Timescales -Port $Port -Range far -ProjectileFixture "$PSScriptRoot/scenarios/base1-flyer-crossing.json" -OutputRoot $OutputRoot
