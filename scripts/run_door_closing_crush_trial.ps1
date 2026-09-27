[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29950,[string]$OutputRoot='')
& "$PSScriptRoot/run_closing_door_phases_trial.ps1" -Timescales $Timescales -Port $Port -OutputRoot $OutputRoot -ReleaseFrames 87
