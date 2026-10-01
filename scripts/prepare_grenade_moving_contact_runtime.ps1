[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=& "$PSScriptRoot/prepare_drop_risk_runtime.ps1" -RuntimeRoot (Join-Path $repo 'workspace/runtime/q2go-grenade-moving-contact')
$entities=@'
{
"classname" "misc_insane"
"origin" "32 -80 24"
"spawnflags" "32"
"target" "grenade_walk_right"
}
{
"classname" "path_corner"
"targetname" "grenade_walk_right"
"origin" "32 -64 24"
"target" "grenade_walk_left"
}
{
"classname" "path_corner"
"targetname" "grenade_walk_left"
"origin" "32 -96 24"
"target" "grenade_walk_right"
}
'@
Add-Content (Join-Path $runtime 'baseq2/maps/base1.ent') ("`n"+$entities) -Encoding ascii
$runtime
