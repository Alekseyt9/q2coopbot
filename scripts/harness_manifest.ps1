# Read-only provenance helpers. Never capture environment values, credentials or file contents.
function Get-HarnessFileRecords {
    param([string]$Root, [string[]]$Paths)
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    foreach ($path in @($Paths | Sort-Object -Unique)) {
        $full = [IO.Path]::GetFullPath($path)
        if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Manifest file is outside root: $full" }
        $file = Get-Item -LiteralPath $full
        [pscustomobject][ordered]@{ path=$full.Substring($prefix.Length).Replace('\','/'); bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
}

function Get-HarnessSourceRecords {
    param([string]$Repository)
    $paths = @()
    foreach ($directory in @('cmd','internal')) {
        $paths += @(Get-ChildItem -LiteralPath (Join-Path $Repository $directory) -Recurse -File -Filter '*.go' | ForEach-Object FullName)
    }
    $paths += @(Get-ChildItem -LiteralPath (Join-Path $Repository 'scripts') -File -Filter '*.ps1' | ForEach-Object FullName)
    foreach ($name in @('go.mod','go.sum')) { $path=Join-Path $Repository $name; if (Test-Path -LiteralPath $path) {$paths += $path} }
    @(Get-HarnessFileRecords -Root $Repository -Paths $paths)
}

function Get-HarnessFingerprint {
    param([object[]]$Records)
    $canonical = @($Records | Sort-Object path | ForEach-Object { $_.path + ':' + $_.sha256 }) -join "`n"
    $sha=[Security.Cryptography.SHA256]::Create()
    try { ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)))).Replace('-','').ToLowerInvariant() }
    finally {$sha.Dispose()}
}
