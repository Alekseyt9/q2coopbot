[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-drop-risk'
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
$maps=@('base1','base3')
$stream=[IO.File]::OpenRead((Join-Path $runtime 'baseq2/pak0.pak'))
$r=[IO.BinaryReader]::new($stream)
try {
    if([Text.Encoding]::ASCII.GetString($r.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$r.ReadInt32();$length=$r.ReadInt32();$stream.Position=$offset;$entries=@{}
    for($i=0;$i -lt $length/64;$i++) {
        $name=[Text.Encoding]::ASCII.GetString($r.ReadBytes(56)).Trim([char]0)
        $at=$r.ReadInt32();$size=$r.ReadInt32();$entries[$name]=@($at,$size)
    }
    foreach($map in $maps) {
        $entry=$entries["maps/$map.bsp"];if(!$entry){throw "Missing BSP $map"}
        $stream.Position=$entry[0];$bsp=$r.ReadBytes($entry[1])
        $ent=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
        $blocks=@([regex]::Matches($ent,'(?s)\{[^{}]*\}')|ForEach-Object Value)
        $filtered=@($blocks | Where-Object {$_ -notmatch '"classname"\s+"(monster_|item_|weapon_|ammo_)'})
        [IO.File]::WriteAllText((Join-Path $runtime "baseq2/maps/$map.ent"),($filtered -join "`n"),[Text.Encoding]::ASCII)
        Copy-Item -LiteralPath (Join-Path $repo "workspace/runtime/q2go/baseq2/maps/$map.aas") -Destination (Join-Path $runtime "baseq2/maps/$map.aas") -Force
    }
} finally {$r.Dispose()}
$runtime

