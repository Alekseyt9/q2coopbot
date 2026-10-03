[CmdletBinding()]
param([Parameter(Mandatory)][string]$RuntimeRoot,[switch]$Blocked,[switch]$LocalChain)
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
"origin" "144 -304 32"
"angle" "180"
}
'@
$a=@'
{
"classname" "func_door"
"model" "*32"
"origin" "1792 -1792 -104"
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
"origin" "-88 16 0"
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
"origin" "0 0 0"
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
if($LocalChain){
    $a=@'
{
"classname" "func_door"
"model" "*32"
"origin" "1792 -1792 -104"
"targetname" "main_lock"
"angle" "-2"
"wait" "-1"
"lip" "0"
"speed" "200"
}
{
"classname" "func_door"
"model" "*33"
"origin" "1872 -1888 -104"
"targetname" "side_lock"
"angle" "-2"
"wait" "-1"
"lip" "0"
"speed" "200"
}
{
"classname" "trigger_once"
"model" "*16"
"origin" "120 32 0"
"target" "main_lock"
}
{
"classname" "trigger_once"
"model" "*13"
"origin" "2160 -1888 -56"
"target" "side_relay"
}
{
"classname" "trigger_relay"
"targetname" "side_relay"
"target" "side_lock"
}
{
"classname" "trigger_multiple"
"model" "*16"
"origin" "-88 16 0"
"target" "finish"
}
{
"classname" "target_changelevel"
"targetname" "finish"
"map" "unit_c"
}
'@
    if($Blocked){$a+="`n{`n`"classname`" `"trigger_always`"`n`"killtarget`" `"side_relay`"`n`"delay`" `"0.2`"`n}`n"}
}
elseif($Blocked){
    # A native startup trigger removes the setter before the client arrives.
    # BSP retains the planned touch chain: an attempted action must not be
    # mistaken for a successful runtime effect.
    $b+="`n{`n`"classname`" `"trigger_always`"`n`"killtarget`" `"unit_set`"`n`"delay`" `"0.2`"`n}`n"
}
foreach($name in @('unit_a','unit_b','unit_c')){
    Copy-Item (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/base1.aas') (Join-Path $RuntimeRoot "baseq2/maps/$name.aas")
    $extra=if($name -eq 'unit_a'){$a}elseif($name -eq 'unit_b'){$b}else{''}
    $mapSpawn=if($name -eq 'unit_b'){$spawn.Replace('144 -304 32','192 -304 32')}else{$spawn}
$text=$world[0].Value+"`n"+$mapSpawn+"`n"+$extra
    # Go reads BSP entities; native Yamagi may use .ent. Keep both identical.
    # Append a replacement entity lump; collision/model/visibility lumps stay
    # byte-for-byte unchanged and use the source geometry's AAS.
    $entityBytes=[Text.Encoding]::ASCII.GetBytes($text+[char]0)
    $fixtureBsp=[byte[]]::new($bsp.Length+$entityBytes.Length)
    [Array]::Copy($bsp,$fixtureBsp,$bsp.Length)
    [Array]::Copy($entityBytes,0,$fixtureBsp,$bsp.Length,$entityBytes.Length)
    [Array]::Copy([BitConverter]::GetBytes([int]$bsp.Length),0,$fixtureBsp,8,4)
    [Array]::Copy([BitConverter]::GetBytes([int]$entityBytes.Length),0,$fixtureBsp,12,4)
    [IO.File]::WriteAllBytes((Join-Path $RuntimeRoot "baseq2/maps/$name.bsp"),$fixtureBsp)
    [IO.File]::WriteAllText((Join-Path $RuntimeRoot "baseq2/maps/$name.ent"),$text,[Text.Encoding]::ASCII)
}
@{version=1;source_map='base1';maps=@('unit_a','unit_b','unit_c');scope='Isolated fixture; reused original BSP/AAS, translated doors and triggers; no monsters/items; fixture spawn';local_chain=[bool]$LocalChain;native_crosslevel_flags=(!$LocalChain);setter_removed_at_runtime=[bool]$Blocked}|ConvertTo-Json|Set-Content (Join-Path $RuntimeRoot 'unit-fixture.json')
$RuntimeRoot

