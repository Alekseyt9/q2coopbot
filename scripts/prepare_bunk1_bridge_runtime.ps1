[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-bunk1-bridge'
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/bunk1.aas') -Destination (Join-Path $runtime 'baseq2/maps/bunk1.aas') -Force
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
        if($name -eq 'maps/bunk1.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'bunk1 missing from pak0'}
    $stream.Position=$entry[0]; $bsp=$reader.ReadBytes($entry[1])
} finally {$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
$blocks=@([regex]::Matches($entities,'(?s)\{[^{}]*\}') | ForEach-Object Value)
foreach($model in @(99)) {
    $door=@($blocks | Where-Object {$_ -match ('"model"\s+"\*'+$model+'"')})
    if($door.Count -ne 1 -or $door[0] -notmatch '"classname"\s+"func_door"' -or $door[0] -notmatch '"targetname"\s+"t97"') {throw "Unexpected bunk1 door$model"}
}
$filtered=@($blocks | Where-Object {$_ -notmatch '"classname"\s+"(monster_|item_|weapon_|ammo_)'})
# The captured player stands between already-open bank doors. Hold this
# observed state; their brushes, movement speed and native use remain intact.
$filtered=@($filtered | ForEach-Object {if($_ -match '"model"\s+"\*(128|129)"') {$_.Replace('}', '"wait" "-1"'+"`n}")} else {$_}})
$filtered+="`n{`n`"classname`" `"trigger_once`"`n`"model`" `"*99`"`n`"origin`" `"0 -250 24`"`n`"target`" `"t97`"`n}`n"
$filtered+="`n{`n`"classname`" `"trigger_once`"`n`"model`" `"*99`"`n`"origin`" `"0 -250 24`"`n`"target`" `"t95`"`n}`n"
$filtered+="`n{`n`"classname`" `"trigger_once`"`n`"model`" `"*99`"`n`"origin`" `"0 -250 24`"`n`"target`" `"t188`"`n}`n"
[IO.File]::WriteAllText((Join-Path $runtime 'baseq2/maps/bunk1.ent'),($filtered -join "`n"),[Text.Encoding]::ASCII)
$runtime

