[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-jail2-door'
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/jail2.aas') -Destination (Join-Path $runtime 'baseq2/maps/jail2.aas') -Force
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
        if($name -eq 'maps/jail2.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'jail2 missing from pak0'}
    $stream.Position=$entry[0]; $bsp=$reader.ReadBytes($entry[1])
} finally {$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
$blocks=@([regex]::Matches($entities,'(?s)\{[^{}]*\}') | ForEach-Object Value)
foreach($model in @(49,50)) {
    $door=@($blocks | Where-Object {$_ -match ('"model"\s+"\*'+$model+'"')})
    if($door.Count -ne 1 -or $door[0] -notmatch '"classname"\s+"func_door"' -or $door[0] -notmatch '"targetname"\s+"t144"') {throw "Unexpected jail2 door$model"}
}
$filtered=@($blocks | Where-Object {$_ -notmatch '"classname"\s+"monster_'})
$filtered+="`n{`n`"classname`" `"trigger_once`"`n`"model`" `"*50`"`n`"origin`" `"0 300 0`"`n`"target`" `"t144`"`n}`n"
[IO.File]::WriteAllText((Join-Path $runtime 'baseq2/maps/jail2.ent'),($filtered -join "`n"),[Text.Encoding]::ASCII)
$runtime
