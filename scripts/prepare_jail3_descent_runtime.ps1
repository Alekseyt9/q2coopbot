[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-jail3-descent'
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
& { param($runtimeSource,$runtimeTarget) . "$PSScriptRoot/prepare_runtime.ps1" -FunctionsOnly; Install-RuntimeImmutable $runtimeSource $runtimeTarget } (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/jail3.aas') (Join-Path $runtime 'baseq2/maps/jail3.aas')
# Keep native brushes, door speed, wait and linked targets. Only remove enemies
# and add a one-shot activation volume around the synchronized actor placement.
$stream=[IO.File]::OpenRead((Join-Path $runtime 'baseq2/pak0.pak'))
$reader=[IO.BinaryReader]::new($stream)
try {
    if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$reader.ReadInt32(); $length=$reader.ReadInt32(); $stream.Position=$offset
    $entry=$null
    for($i=0;$i -lt $length/64;$i++) {
        $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0)
        $at=$reader.ReadInt32(); $size=$reader.ReadInt32()
        if($name -eq 'maps/jail3.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'jail3 missing from pak0'}
    $stream.Position=$entry[0]; $bsp=$reader.ReadBytes($entry[1])
} finally {$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
$blocks=@([regex]::Matches($entities,'(?s)\{[^{}]*\}') | ForEach-Object Value)
# Preserve native geometry and mechanisms; isolate navigation from combat/items.
$filtered=@($blocks | Where-Object {$_ -notmatch '"classname"\s+"(monster_|item_|weapon_|ammo_)'})
[IO.File]::WriteAllText((Join-Path $runtime 'baseq2/maps/jail3.ent'),($filtered -join "`n"),[Text.Encoding]::ASCII)
$runtime
