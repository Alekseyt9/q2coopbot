# Requires PowerShell 7. Each compiler has an isolated working directory.
[CmdletBinding()]
param(
    [string]$AssetsRoot='F:/src/quake2/assets/baseq2',
    [string]$OutputRoot='',
    [ValidateRange(1,8)][int]$Parallelism=4,
    [int]$WallLimitSeconds=900
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/aas-batch-'+(Get-Date -Format yyyyMMdd-HHmmss))}
if(Test-Path $OutputRoot){throw "Output must be new: $OutputRoot"}
$inputs=Join-Path $OutputRoot 'bsp'
New-Item -ItemType Directory $inputs -Force | Out-Null
$inputs=(Resolve-Path $inputs).Path
$OutputRoot=(Resolve-Path $OutputRoot).Path
# Later numbered PAKs override earlier PAKs; loose maps override archives.
foreach($pak in Get-ChildItem $AssetsRoot -Filter 'pak*.pak' | Sort-Object { [int]($_.BaseName -replace '^pak','') }) {
    $stream=[IO.File]::OpenRead($pak.FullName);$reader=[IO.BinaryReader]::new($stream)
    try {
        if([Text.Encoding]::ASCII.GetString($reader.ReadBytes(4)) -ne 'PACK'){throw "Invalid PAK: $pak"}
        $offset=$reader.ReadInt32();$length=$reader.ReadInt32()
        if($offset -lt 12 -or $length -lt 0 -or $length%64 -ne 0 -or $offset+$length -gt $stream.Length){throw 'Invalid PAK directory'}
        $entries=@();$stream.Position=$offset
        for($i=0;$i -lt $length/64;$i++) {
            $name=[Text.Encoding]::ASCII.GetString($reader.ReadBytes(56)).Trim([char]0)
            $at=$reader.ReadInt32();$size=$reader.ReadInt32()
            if($name -match '^maps/([A-Za-z0-9_]+)\.bsp$') {
                if($at -lt 0 -or $size -lt 0 -or [long]$at+$size -gt $stream.Length){throw "Invalid entry $name"}
                $entries+=@{name=$Matches[1];at=$at;size=$size}
            }
        }
        foreach($entry in $entries){$stream.Position=$entry.at;[IO.File]::WriteAllBytes((Join-Path $inputs ($entry.name+'.bsp')),$reader.ReadBytes($entry.size))}
    } finally {$reader.Dispose()}
}
if(Test-Path (Join-Path $AssetsRoot 'maps')) {
    Get-ChildItem (Join-Path $AssetsRoot 'maps') -Filter '*.bsp' | ForEach-Object {
        if($_.BaseName -notmatch '^[A-Za-z0-9_]+$'){throw 'Invalid loose map name'}
        Copy-Item $_.FullName $inputs -Force
    }
}
$maps=@(Get-ChildItem $inputs -Filter '*.bsp')
if(!$maps.Count){throw 'No BSP maps found'}
$compiler=Join-Path $repo 'workspace/build/aas-current/workspace/tools/bspc/bspc.exe'
$compile=Join-Path $PSScriptRoot 'compile_aas.ps1'
@{maps=@($maps.BaseName);parallelism=$Parallelism;compiler_sha256=(Get-FileHash $compiler).Hash}|ConvertTo-Json -Depth 4|Set-Content (Join-Path $OutputRoot 'manifest.json')
$results=@($maps | ForEach-Object -ThrottleLimit $Parallelism -Parallel {
    $ErrorActionPreference='Stop'
    $out=Join-Path $using:OutputRoot $_.BaseName
    try {
        $summary=& $using:compile -BspPath $_.FullName -BspcExe $using:compiler -OutputRoot $out -Threads 1 -WallLimitSeconds $using:WallLimitSeconds
        [pscustomobject]@{map=$_.BaseName;accepted=$true;summary=$summary;error=''}
    } catch {
        [pscustomobject]@{map=$_.BaseName;accepted=$false;summary=$null;error=$_.Exception.Message}
    }
})
$results | Sort-Object map | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputRoot 'report.json')
$destination=Join-Path $OutputRoot 'maps';New-Item -ItemType Directory $destination | Out-Null
foreach($result in $results | Where-Object accepted){Copy-Item -LiteralPath $result.summary.aas -Destination $destination}
$results | Select-Object map,accepted,error | Format-Table
Write-Output "AAS batch: $OutputRoot"
if(@($results | Where-Object { !$_.accepted }).Count){throw 'Some maps failed; see report.json. Successful outputs retained.'}
