$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'read_episode_registry.ps1')
$repo = Split-Path -Parent $PSScriptRoot
$actual = Read-EpisodeRegistry (Join-Path $repo 'scripts/scenarios/episodes/index.json')
if (!$actual.episodes.Count) { throw 'Empty registry' }
$root = Join-Path $repo ('workspace/artifacts/registry-test-' + [guid]::NewGuid().ToString('N'))
foreach ($case in @('valid','duplicate-id','duplicate-file','missing-file','unlisted','outside')) {
    $dir = Join-Path $root $case
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    '{"id":"one"}' | Set-Content (Join-Path $dir 'one.json')
    $files = @('one.json')
    switch ($case) {
        'duplicate-id' { '{"id":"one"}' | Set-Content (Join-Path $dir 'two.json'); $files += 'two.json' }
        'duplicate-file' { $files += 'one.json' }
        'missing-file' { $files += 'missing.json' }
        'unlisted' { '{"id":"two"}' | Set-Content (Join-Path $dir 'two.json') }
        'outside' { $files = @('../one.json') }
    }
    @{version=2;files=$files} | ConvertTo-Json | Set-Content (Join-Path $dir 'index.json')
    $failed = $false
    try { $null = Read-EpisodeRegistry (Join-Path $dir 'index.json') } catch { $failed = $true }
    if ($failed -ne ($case -ne 'valid')) { throw "Unexpected registry result: $case" }
}
Write-Host "PASS: $($actual.episodes.Count) episodes loaded; duplicate/missing/unlisted/outside files rejected"
