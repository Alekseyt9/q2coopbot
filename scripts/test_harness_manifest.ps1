$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'harness_manifest.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$dir=Join-Path $repo ('workspace/artifacts/manifest-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $dir | Out-Null
$file=Join-Path $dir 'input.txt'
[IO.File]::WriteAllText($file,'before')
$records=@(Get-HarnessFileRecords -Root $dir -Paths @($file))
if ($records.Count -ne 1 -or $records[0].path -ne 'input.txt') {throw 'Relative path or file count incorrect.'}
$before=Get-HarnessFingerprint -Records $records
[IO.File]::WriteAllText($file,'after')
$after=Get-HarnessFingerprint -Records @(Get-HarnessFileRecords -Root $dir -Paths @($file))
if ($before -eq $after) {throw 'Content mutation not detected.'}
$a=@([pscustomobject]@{path='a';sha256='1'},[pscustomobject]@{path='b';sha256='2'})
if ((Get-HarnessFingerprint $a) -ne (Get-HarnessFingerprint @($a[1],$a[0]))) {throw 'Input ordering affected fingerprint.'}
$rejected=$false
try { Get-HarnessFileRecords -Root $dir -Paths @((Join-Path $repo 'go.mod')) | Out-Null } catch {$rejected=$true}
if (-not $rejected) {throw 'Out-of-root file accepted.'}
Write-Output 'Manifest tests passed: paths, content changes, ordering, root boundary.'
