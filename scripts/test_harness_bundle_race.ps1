[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$key=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes('bundle-race-test-'+[guid]::NewGuid().ToString('N')))).ToLowerInvariant()
$results=1,2|ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    $repo=$using:repo
    . "$repo/scripts/harness_manifest.ps1"
    . "$repo/scripts/combat_harness_bundle.ps1"
    # A unique test-only cache key forces cold publication without mutating or
    # removing any production cache entry. Actual Go binaries are built.
    $script:raceTestKey=$using:key
    function Get-HarnessFingerprint {param($Records);return $script:raceTestKey}
    $bundle=New-HarnessBinaryBundle $repo
    $null=Assert-HarnessBinaryBundle $bundle $script:raceTestKey
    [pscustomobject]@{bundle=$bundle;verified=$true}
} -ThrottleLimit 2
if(@($results).Count -ne 2 -or @($results.bundle|Select-Object -Unique).Count -ne 1){throw 'Concurrent builders did not share one verified winner'}
Write-Output "PASS: two cold concurrent builders share one immutable verified bundle ($($results[0].bundle))"
