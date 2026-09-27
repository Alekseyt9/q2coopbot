$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-base3-pickup'
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base3 -RuntimeRoot $runtime | Out-Null
# Recreate the observed dropped shotgun as a stationary native pickup.
# Shell boxes and collision remain from the original map; combat is excluded.
@'
{
"classname" "weapon_shotgun"
"origin" "1168 752 -800"
}
'@ | Add-Content (Join-Path $runtime 'baseq2/maps/base3.ent')
$runtime
