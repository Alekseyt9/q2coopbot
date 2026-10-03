[CmdletBinding()]
param([Parameter(Mandatory)][string]$RuntimeRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(Test-Path $RuntimeRoot){throw 'Fresh unit fixture runtime required'}
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $RuntimeRoot | Out-Null
$reader=[IO.BinaryReader]::new([IO.File]::OpenRead((Join-Path $RuntimeRoot 'baseq2/pak0.pak')))
try{
    if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$reader.ReadInt32();$length=$reader.ReadInt32();$reader.BaseStream.Position=$offset
    $entry=$null
    for($i=0;$i -lt $length/64;$i++){
        $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0)
        $at=$reader.ReadInt32();$size=$reader.ReadInt32()
        if($name -eq 'maps/base1.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'base1 BSP absent'}
    $reader.BaseStream.Position=$entry[0];$bsp=$reader.ReadBytes($entry[1])
}finally{$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12))
$world=@([regex]::Matches($entities,'(?s)\{[^{}]*\}')|Where-Object {$_.Value -match '"classname"\s+"worldspawn"'})
if($world.Count -ne 1){throw 'Worldspawn absent/ambiguous'}
$spawn=@'
{
"classname" "info_player_start"
"origin" "144 -384 32"
"angle" "180"
}
'@
$a=@'
{
"classname" "func_door"
"model" "*32"
"origin" "1792 -1840 -104"
"targetname" "unit_lock"
"angle" "-2"
"wait" "-1"
"lip" "0"
"speed" "200"
}
{
"classname" "target_crosslevel_target"
"spawnflags" "1"
"target" "unit_lock"
"delay" "0.3"
}
{
"classname" "trigger_multiple"
"model" "*16"
"origin" "-64 -64 0"
"target" "finish"
}
{
"classname" "target_changelevel"
"targetname" "finish"
"map" "unit_c"
}
{
"classname" "trigger_multiple"
"model" "*8"
"origin" "0 -160 0"
"target" "remote"
}
{
"classname" "target_changelevel"
"targetname" "remote"
"map" "unit_b"
}
'@
$b=@'
{
"classname" "trigger_once"
"model" "*16"
"target" "unit_set"
}
{
"classname" "target_crosslevel_trigger"
"spawnflags" "1"
"targetname" "unit_set"
}
{
"classname" "trigger_multiple"
"model" "*8"
"target" "return"
}
{
"classname" "target_changelevel"
"targetname" "return"
"map" "unit_a"
}
'@
foreach($name in @('unit_a','unit_b','unit_c')){
    [IO.File]::WriteAllBytes((Join-Path $RuntimeRoot "baseq2/maps/$name.bsp"),$bsp)
    Copy-Item (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/base1.aas') (Join-Path $RuntimeRoot "baseq2/maps/$name.aas")
    $extra=if($name -eq 'unit_a'){$a}elseif($name -eq 'unit_b'){$b}else{''}
    [IO.File]::WriteAllText((Join-Path $RuntimeRoot "baseq2/maps/$name.ent"),($world[0].Value+"`n"+$spawn+"`n"+$extra),[Text.Encoding]::ASCII)
}
@{version=1;source_map='base1';maps=@('unit_a','unit_b','unit_c');scope='Isolated cross-level fixture; reused original BSP/AAS, translated door and triggers; no monsters/items; fixture spawn';native_crosslevel_flags=$true}|ConvertTo-Json|Set-Content (Join-Path $RuntimeRoot 'unit-fixture.json')
$RuntimeRoot
