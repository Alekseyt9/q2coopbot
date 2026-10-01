[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=& "$PSScriptRoot/prepare_drop_risk_runtime.ps1" -RuntimeRoot (Join-Path $repo 'workspace/runtime/q2go-grenade-contact')
$entity=@'
{
"classname" "misc_insane"
"origin" "-680 1141 296"
"spawnflags" "24"
"angle" "180"
}
{
"classname" "misc_insane"
"origin" "-680 1149 296"
"spawnflags" "24"
"angle" "180"
}
'@
Add-Content (Join-Path $runtime 'baseq2/maps/base1.ent') ("`n"+$entity) -Encoding ascii
$runtime
