[CmdletBinding()]
param([ValidateSet('base2','base3')][string]$Map='base3',[switch]$SideWalls,[string]$RuntimeRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo ('workspace/runtime/q2go-elevator-cycle'+$(if($Map -eq 'base2'){'-base2'}))
if($SideWalls){
    if($Map -ne 'base2'){throw 'Side walls fixture requires base2'}
    $runtime+='-side-walls'
}
if($RuntimeRoot){$runtime=$RuntimeRoot}
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
Copy-Item -LiteralPath (Join-Path $repo "workspace/runtime/q2go/baseq2/maps/$Map.aas") -Destination (Join-Path $runtime "baseq2/maps/$Map.aas") -Force
# Extract only the entity lump. The BSP, collision and native mover physics stay unchanged.
$stream=[IO.File]::OpenRead((Join-Path $runtime 'baseq2/pak0.pak'))
$reader=[IO.BinaryReader]::new($stream)
try {
    if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$reader.ReadInt32(); $length=$reader.ReadInt32(); $stream.Position=$offset
    $entry=$null
    for($i=0;$i -lt $length/64;$i++) {
        $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0)
        $at=$reader.ReadInt32(); $size=$reader.ReadInt32()
        if($name -eq "maps/$Map.bsp"){$entry=@($at,$size)}
    }
    if(!$entry){throw "$Map missing from pak0"}
    $stream.Position=$entry[0]; $bsp=$reader.ReadBytes($entry[1])
} finally {$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
# Remove combat actors only; keep world geometry, triggers and platform physics.
$monsters=@([regex]::Matches($entities,'(?s)\{[^{}]*\}')|Where-Object {$_.Value -match '"classname"\s+"monster_'})
if(!$monsters.Count){throw 'No monsters found in fixture source'}
foreach($monster in $monsters){$entities=$entities.Replace($monster.Value,'')}
if($SideWalls){
    # Reuse solid BSP model1 as two visible walls. Translated bounds:
    # x[-64,64], y[1372,1380]/[1436,1444], z[0,56].
    # The central 56-unit corridor and native lift remain open.
    foreach($y in @(-4,60)){
        $entities+="`n{`n`"classname`" `"func_wall`"`n`"model`" `"*1`"`n`"origin`" `"-576 $y -24`"`n}`n"
    }
}
[IO.File]::WriteAllText((Join-Path $runtime "baseq2/maps/$Map.ent"),$entities,[Text.Encoding]::ASCII)
@{map=$Map;removed_monsters=$monsters.Count;native_platform_physics=$true;side_walls=[bool]$SideWalls;scope='navigation_without_combat'}|ConvertTo-Json|Set-Content (Join-Path $runtime 'elevator-fixture.json')
$runtime
