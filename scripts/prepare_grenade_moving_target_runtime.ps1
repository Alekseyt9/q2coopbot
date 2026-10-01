[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=& "$PSScriptRoot/prepare_drop_risk_runtime.ps1" -RuntimeRoot (Join-Path $repo 'workspace/runtime/q2go-grenade-moving-target')
$entities=@'
{
"classname" "misc_insane"
"origin" "128 -128 24"
"spawnflags" "32"
"angle" "270"
"target" "grenade_cross_down"
}
{
"classname" "path_corner"
"targetname" "grenade_cross_down"
"origin" "128 -304 24"
"target" "grenade_cross_up"
}
{
"classname" "path_corner"
"targetname" "grenade_cross_up"
"origin" "128 -128 24"
"target" "grenade_cross_down"
}
'@
Add-Content (Join-Path $runtime 'baseq2/maps/base1.ent') ("`n"+$entities) -Encoding ascii
$runtime
