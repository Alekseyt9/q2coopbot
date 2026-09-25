[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BspPath,
    [string]$BspcExe = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\build\aas\workspace\tools\bspc\bspc.exe'),
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) ('workspace\artifacts\aas-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [int]$Threads = 1,
    [int]$WallLimitSeconds = 900
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $BspPath -PathType Leaf)) { throw "BSP file is missing: $BspPath" }
if (-not (Test-Path -LiteralPath $BspcExe -PathType Leaf)) { throw "BSPC is missing: $BspcExe" }
if ($Threads -lt 1 -or $WallLimitSeconds -lt 1) { throw 'Threads and wall limit must be positive.' }
$map = [IO.Path]::GetFileNameWithoutExtension($BspPath)
if ($map -notmatch '^[A-Za-z0-9_]+$') { throw "Invalid map name: $map" }
$bsp = (Resolve-Path -LiteralPath $BspPath).Path
$compiler = (Resolve-Path -LiteralPath $BspcExe).Path
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$output = (Resolve-Path -LiteralPath $OutputRoot).Path
$aas = Join-Path $output "$map.aas"
if (Test-Path -LiteralPath $aas) { throw "Output already exists: $aas" }
$stdout = Join-Path $output "$map-bspc.log"
$stderr = Join-Path $output "$map-bspc.err.log"
$args = @('-bsp2aas', $bsp, '-output', $output, '-threads', "$Threads")
$process = Start-Process -FilePath $compiler -ArgumentList $args -WorkingDirectory $output `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
try {
    if (-not $process.WaitForExit($WallLimitSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force
        throw "BSPC timed out after $WallLimitSeconds seconds: $stdout"
    }
    if ($process.ExitCode -ne 0) { throw "BSPC exited with code $($process.ExitCode): $stdout" }
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}
if (-not (Test-Path -LiteralPath $aas -PathType Leaf)) { throw "BSPC did not create AAS: $stdout" }
$data = [IO.File]::ReadAllBytes($aas)
if ($data.Length -lt 124 -or [Text.Encoding]::ASCII.GetString($data, 0, 4) -ne 'EAAS' -or
    [BitConverter]::ToInt32($data, 4) -ne 5) {
    throw "BSPC produced an unsupported AAS: $aas"
}
$header = [byte[]]$data[0..123]
for ($i = 8; $i -lt 124; $i++) {
    $header[$i] = $header[$i] -bxor [byte]((($i - 8) * 119) -band 255)
}
$counts = @{}
foreach ($lump in @(@(7, 48, 'areas'), @(8, 28, 'settings'), @(9, 44, 'reachabilities'))) {
    $index, $rowSize, $name = $lump
    $offset = [BitConverter]::ToInt32($header, 12 + $index * 8)
    $length = [BitConverter]::ToInt32($header, 16 + $index * 8)
    if ($offset -lt 124 -or $length -le 0 -or $length % $rowSize -ne 0 -or
        $offset -gt $data.Length -or $length -gt $data.Length - $offset) {
        throw "Invalid $name lump in $aas"
    }
    $counts[$name] = [int]($length / $rowSize)
}
if ($counts.areas -ne $counts.settings) { throw "AAS area/settings mismatch: $aas" }
$summary = [pscustomobject]@{
    map = $map; aas = $aas; version = 5; areas = $counts.areas
    reachabilities = $counts.reachabilities; sha256 = (Get-FileHash -LiteralPath $aas).Hash
    compiler_log = $stdout
}
$summaryPath = Join-Path $output "$map-summary.json"
$summary | ConvertTo-Json | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Output $summary
