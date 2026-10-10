[CmdletBinding()]
param([Parameter(Mandatory)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=[IO.Path]::GetFullPath($OutputRoot)
$allowed=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts')).TrimEnd('\')+'\'
if(!$root.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $root)){throw 'Fresh artifacts test root required'}
New-Item -ItemType Directory -Path $root|Out-Null
$source=Join-Path $root 'test-binary.dat'
[IO.File]::WriteAllBytes($source,[Text.Encoding]::UTF8.GetBytes('immutable binary shard capacity test'))
$digest=(Get-FileHash -LiteralPath $source).Hash.ToLowerInvariant()
# 1050 links exceeds the real NTFS limit, with competing creators.
0..1049|ForEach-Object -Parallel {
    . "$using:repo/scripts/combat_harness_bundle.ps1"
    Install-HarnessBinaryLink $using:source (Join-Path $using:root ('link-'+$_+'.dat')) $using:digest
} -ThrottleLimit 16
$links=@(Get-ChildItem -LiteralPath $root -Filter 'link-*.dat')
$shards=@(Get-ChildItem -LiteralPath (Join-Path $root 'link-shards') -File)
if($links.Count -ne 1050 -or $shards.Count -ne 1){throw 'Shard count or installed links differ'}
foreach($link in $links){if((Get-FileHash -LiteralPath $link.FullName).Hash.ToLowerInvariant() -ne $digest){throw 'Installed bytes changed'}}
@{state='complete';links=1050;extra_copies=$shards.Count;sha256=$digest;parallelism=16}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root 'report.json') -Encoding utf8NoBOM
