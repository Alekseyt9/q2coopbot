[CmdletBinding()]
param([Parameter(Mandatory)][string]$RuntimeRoot,[switch]$Blocked,[switch]$Shoot)
$ErrorActionPreference='Stop'
$entityPath=Join-Path $RuntimeRoot 'baseq2/maps/base2.ent'
$text=[IO.File]::ReadAllText($entityPath)
$buttonEntities=@([regex]::Matches($text,'(?s)\{[^{}]*\}')|Where-Object {$_.Value -match '"classname"\s+"func_button"' -and $_.Value -match '"model"\s+"\*34"'})
if($buttonEntities.Count -ne 1 -or $buttonEntities[0].Value -notmatch '"target"\s+"t29"'){throw 'Original button34 target absent'}
$replacement=[regex]::Replace($buttonEntities[0].Value,'"target"\s+"t29"','"target" "test_button_relay"')
if($Shoot){$replacement=$replacement.TrimEnd('}')+"`n`"health`" `"10`"`n}"}
$text=$text.Replace($buttonEntities[0].Value,$replacement)
$text+="`n{`n`"classname`" `"trigger_relay`"`n`"targetname`" `"test_button_relay`"`n`"target`" `"t29`"`n}`n"
if($Blocked){$text+="`n{`n`"classname`" `"trigger_always`"`n`"killtarget`" `"test_button_relay`"`n`"delay`" `"0.2`"`n}`n"}
$reader=[IO.BinaryReader]::new([IO.File]::OpenRead((Join-Path $RuntimeRoot 'baseq2/pak0.pak')))
try{
    if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw 'Invalid PAK'}
    $offset=$reader.ReadInt32();$length=$reader.ReadInt32();$reader.BaseStream.Position=$offset;$entry=$null
    for($i=0;$i -lt $length/64;$i++){
        $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0);$at=$reader.ReadInt32();$size=$reader.ReadInt32()
        if($name -eq 'maps/base2.bsp'){$entry=@($at,$size)}
    }
    if(!$entry){throw 'Original BSP absent'}
    $reader.BaseStream.Position=$entry[0];$bsp=$reader.ReadBytes($entry[1])
}finally{$reader.Dispose()}
# Append only entities. Geometry/model/collision/visibility lumps stay intact.
$bytes=[Text.Encoding]::ASCII.GetBytes($text+[char]0)
$fixture=[byte[]]::new($bsp.Length+$bytes.Length)
[Array]::Copy($bsp,$fixture,$bsp.Length);[Array]::Copy($bytes,0,$fixture,$bsp.Length,$bytes.Length)
[Array]::Copy([BitConverter]::GetBytes([int]$bsp.Length),0,$fixture,8,4);[Array]::Copy([BitConverter]::GetBytes([int]$bytes.Length),0,$fixture,12,4)
[IO.File]::WriteAllBytes((Join-Path $RuntimeRoot 'baseq2/maps/base2.bsp'),$fixture)
[IO.File]::WriteAllText($entityPath,$text,[Text.Encoding]::ASCII)
@{scope='original_geometry_modified_button_link';button_model=34;door_model=33;relay='test_button_relay';shoot=[bool]$Shoot;native_removal=[bool]$Blocked;collision_unchanged=$true}|ConvertTo-Json|Set-Content (Join-Path $RuntimeRoot 'button-relay-fixture.json')
