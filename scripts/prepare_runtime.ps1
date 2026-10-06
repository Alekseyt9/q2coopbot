[CmdletBinding()]
param(
    [string]$AssetsRoot = 'F:\src\quake2\assets\baseq2',
    [string]$AASRoot = '',
    [string]$ServerExe = 'F:\src\quake2\yquake2\build\codex-speed-test\release\q2ded.exe',
    [string]$GameDll = 'F:\src\quake2\yquake2\build\codex-speed-test\release\baseq2\game.dll',
    [string]$RuntimeRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace\runtime\q2go'),
    [switch]$FunctionsOnly
)

$ErrorActionPreference = 'Stop'
# Replace the directory entry, not the contents of a potentially shared hard link.
# Live sessions may still be using the previous AAS through another path.
function Get-RuntimeImmutableHash([string]$Path) {
    for ($attempt = 0; ; $attempt++) {
        $stream = $null
        $algorithm = $null
        try {
            # Readers must permit another worker to atomically replace its link.
            $sharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
            $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, $sharing)
            $algorithm = [Security.Cryptography.SHA256]::Create()
            return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        } catch [IO.IOException] {
            if ($attempt -ge 100) { throw }
            Start-Sleep -Milliseconds 50
        } finally {
            if ($algorithm) { $algorithm.Dispose() }
            if ($stream) { $stream.Dispose() }
        }
    }
}
function Install-RuntimeImmutable([string]$Source, [string]$Target, [string]$PoolRoot = '') {
    $temporary = $Target + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        # Keep a content-addressed snapshot: build outputs and source AAS may change.
        $digest = Get-RuntimeImmutableHash $Source
        if ((Test-Path -LiteralPath $Target) -and
            (Get-RuntimeImmutableHash $Target) -eq $digest) {
            return
        }
        if (-not $PoolRoot) {
            $PoolRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace/build/runtime-immutable'
        }
        $snapshotDir = Join-Path $poolRoot $digest
        New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
        $snapshot = Join-Path $snapshotDir 'source.bin'
        if (-not (Test-Path -LiteralPath $snapshot)) {
            $pending = Join-Path $snapshotDir ([guid]::NewGuid().ToString('N') + '.tmp')
            try {
                Copy-Item -LiteralPath $Source -Destination $pending
                if ((Get-RuntimeImmutableHash $pending) -ne $digest) {
                    throw "Source changed while preparing immutable runtime file: $Source"
                }
                try { [IO.File]::Move($pending, $snapshot) } catch {
                    if (-not (Test-Path -LiteralPath $snapshot)) { throw }
                }
            } finally {
                if (Test-Path -LiteralPath $pending) { Remove-Item -LiteralPath $pending }
            }
        }
        if ((Get-RuntimeImmutableHash $snapshot) -ne $digest) {
            throw "Immutable runtime snapshot differs from source: $snapshot"
        }
        if ([IO.Path]::GetPathRoot($snapshot) -eq [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Target))) {
            Install-RuntimePak $snapshot $temporary $poolRoot
        } else {
            Copy-Item -LiteralPath $snapshot -Destination $temporary
        }
        if (Test-Path -LiteralPath $Target) {
            # Another worker can briefly hash the shared old snapshot without
            # FILE_SHARE_DELETE. Retry the atomic replacement, never overwrite it.
            for ($attempt = 0; ; $attempt++) {
                try {
                    [IO.File]::Replace($temporary, $Target, [NullString]::Value)
                    break
                } catch [IO.IOException] {
                    if ($attempt -ge 100) { throw }
                    Start-Sleep -Milliseconds 50
                }
            }
        } else {
            [IO.File]::Move($temporary, $Target)
        }
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}
# NTFS permits only 1024 links per file. Shared immutable PAK shards avoid a
# private 184MB copy per episode once the original archive reaches that limit.
function Install-RuntimePak([string]$Source, [string]$Target, [string]$PoolRoot) {
    try {
        New-Item -ItemType HardLink -Path $Target -Target $Source -ErrorAction Stop | Out-Null
        return
    } catch {
        if (Test-Path -LiteralPath $Target) { throw }
    }
    $digest = Get-RuntimeImmutableHash $Source
    $pool = Join-Path $PoolRoot $digest
    New-Item -ItemType Directory -Path $pool -Force | Out-Null
    foreach ($index in 0..31) {
        $anchor = Join-Path $pool "asset-$index.pak"
        if (-not (Test-Path -LiteralPath $anchor)) {
            $temporary = Join-Path $pool ([guid]::NewGuid().ToString('N') + '.tmp')
            try {
                Copy-Item -LiteralPath $Source -Destination $temporary
                try { [IO.File]::Move($temporary, $anchor) } catch {
                    if (-not (Test-Path -LiteralPath $anchor)) { throw }
                }
            } finally {
                if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
            }
        }
        if ((Get-RuntimeImmutableHash $anchor) -ne $digest) {
            throw "Immutable PAK pool differs from source: $anchor"
        }
        try {
            New-Item -ItemType HardLink -Path $Target -Target $anchor -ErrorAction Stop | Out-Null
            return
        } catch {
            if (Test-Path -LiteralPath $Target) { throw }
        }
    }
    throw "Cannot link PAK from bounded shared pool: $Source"
}
if ($FunctionsOnly) { return }
foreach ($path in @($AssetsRoot, $ServerExe, $GameDll)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required input is missing: $path" }
}
if ($AASRoot -and -not (Test-Path -LiteralPath $AASRoot -PathType Container)) {
    throw "AAS directory is missing: $AASRoot"
}
$additionalAAS = @()
if ($AASRoot) {
    $additionalAAS = @(Get-ChildItem -LiteralPath $AASRoot -Filter '*.aas' -File)
    if ($additionalAAS.Count -eq 0) { throw "No AAS files found in: $AASRoot" }
    foreach ($aas in $additionalAAS) {
        $data = [IO.File]::ReadAllBytes($aas.FullName)
        if ($data.Length -lt 120 -or [Text.Encoding]::ASCII.GetString($data, 0, 4) -ne 'EAAS') {
            throw "Invalid AAS header: $($aas.FullName)"
        }
        $version = [BitConverter]::ToInt32($data, 4)
        if ($version -notin @(2, 3, 4, 5)) { throw "Unsupported AAS version $version`: $($aas.FullName)" }
        $headerSize = if ($version -ge 4) { 124 } else { 120 }
        if ($data.Length -lt $headerSize) { throw "Truncated AAS header: $($aas.FullName)" }
        $header = [byte[]]$data[0..($headerSize - 1)]
        if ($version -eq 5) {
            for ($i = 8; $i -lt $headerSize; $i++) {
                $header[$i] = $header[$i] -bxor [byte]((($i - 8) * 119) -band 255)
            }
        }
        $reachDirectory = $headerSize - 14 * 8 + 9 * 8
        $reachOffset = [BitConverter]::ToInt32($header, $reachDirectory)
        $reachLength = [BitConverter]::ToInt32($header, $reachDirectory + 4)
        if (
            $reachLength -le 0 -or $reachLength % 44 -ne 0 -or $reachOffset -lt $headerSize -or
            $reachOffset -gt $data.Length -or $reachLength -gt $data.Length - $reachOffset) {
            throw "AAS has no reachability data: $($aas.FullName)"
        }
    }
}
$marker = Join-Path $RuntimeRoot '.q2go-prepared-runtime'
if ((Test-Path -LiteralPath $RuntimeRoot) -and -not (Test-Path -LiteralPath $marker)) {
    throw "Target already exists and was not created by this script: $RuntimeRoot"
}
$baseq2 = Join-Path $RuntimeRoot 'baseq2'
$maps = Join-Path $baseq2 'maps'
New-Item -ItemType Directory -Path $maps -Force | Out-Null
foreach ($name in @('pak0.pak', 'pak1.pak', 'pak2.pak')) {
    $source = Join-Path $AssetsRoot $name
    if (-not (Test-Path -LiteralPath $source)) { continue }
    $target = Join-Path $baseq2 $name
    if (Test-Path -LiteralPath $target) { continue }
    if ([IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $source).Path) -eq [IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $RuntimeRoot).Path)) {
        $poolRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'workspace/build/runtime-paks'
        Install-RuntimePak $source $target $poolRoot
    } else {
        Copy-Item -LiteralPath $source -Destination $target
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $baseq2 'pak0.pak'))) { throw 'pak0.pak is required.' }
$sourceMaps = Join-Path $AssetsRoot 'maps'
if (Test-Path -LiteralPath $sourceMaps) {
    Get-ChildItem -LiteralPath $sourceMaps -Filter '*.aas' -File | ForEach-Object {
        $target = Join-Path $maps $_.Name
        # Asset archives can contain older graphs. Updates must use explicit AASRoot.
        if (-not (Test-Path -LiteralPath $target)) {
            Install-RuntimeImmutable $_.FullName $target
        }
    }
}
foreach ($aas in $additionalAAS) {
    Install-RuntimeImmutable $aas.FullName (Join-Path $maps $aas.Name)
}
Install-RuntimeImmutable $ServerExe (Join-Path $RuntimeRoot 'q2ded.exe')
Install-RuntimeImmutable $GameDll (Join-Path $baseq2 'game.dll')
Set-Content -LiteralPath $marker -Value 'Go UDP test runtime; generated by scripts/prepare_runtime.ps1' -Encoding UTF8
Write-Output "Prepared $RuntimeRoot"
