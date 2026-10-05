[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-closing-door'
& "$PSScriptRoot/prepare_runtime.ps1" -RuntimeRoot $runtime | Out-Null
& { param($runtimeSource,$runtimeTarget) . "$PSScriptRoot/prepare_runtime.ps1" -FunctionsOnly; Install-RuntimeImmutable $runtimeSource $runtimeTarget } (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/base2.aas') (Join-Path $runtime 'baseq2/maps/base2.aas')
# Extract only the entity lump. The BSP, collision and native door physics stay unchanged.
$stream=[IO.File]::OpenRead((Join-Path $runtime 'baseq2/pak0.pak'))
$reader=[IO.BinaryReader]::new($stream)
try {
    if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$reader.ReadInt32(); $length=$reader.ReadInt32(); $stream.Position=$offset
    $entry=$null
    for($i=0;$i -lt $length/64;$i++) {
        $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0)
        $at=$reader.ReadInt32(); $size=$reader.ReadInt32()
        if($name -eq 'maps/base2.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'base2 missing from pak0'}
    $stream.Position=$entry[0]; $bsp=$reader.ReadBytes($entry[1])
} finally {$reader.Dispose()}
$entities=[Text.Encoding]::ASCII.GetString($bsp,[BitConverter]::ToInt32($bsp,8),[BitConverter]::ToInt32($bsp,12)).Trim([char]0)
$doors=@([regex]::Matches($entities,'(?s)\{[^{}]*\}') | Where-Object {$_.Value -match '"model"\s+"\*27"'})
if($doors.Count -ne 1 -or $doors[0].Value -notmatch '"classname"\s+"func_door"' -or $doors[0].Value -match '"targetname"'){throw 'Unexpected base2 door27 definition'}
$replacement=$doors[0].Value.Replace('}', '"targetname" "harness_closing_door"'+"`n}")
$entities=$entities.Replace($doors[0].Value,$replacement)
# A one-shot volume is touched by the placed actor at synchronized frame40.
# Unlike a map-start timer, this cannot drift with connection startup time.
$entities+="`n{`n`"classname`" `"trigger_once`"`n`"model`" `"*27`"`n`"origin`" `"-64 0 0`"`n`"target`" `"harness_closing_door`"`n}`n"
[IO.File]::WriteAllText((Join-Path $runtime 'baseq2/maps/base2.ent'),$entities,[Text.Encoding]::ASCII)
$modelOffset=[BitConverter]::ToInt32($bsp,8+13*8)+27*48
$minX=[BitConverter]::ToSingle($bsp,$modelOffset)
[pscustomobject]@{map='base2';model=27;door_min_x=$minX;standing_hull_boundary_x=$minX-16;opening_trigger='actor_once';native_door_physics=$true;proximity_trigger=$false} |
    ConvertTo-Json | Set-Content (Join-Path $runtime 'door-fixture.json')
$runtime
