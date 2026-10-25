[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/prepare_runtime.ps1" -FunctionsOnly
$root=Join-Path $repo ('workspace/artifacts/diagnostic-engine-manifest-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $root 'baseq2')|Out-Null
Set-Content -LiteralPath (Join-Path $root 'q2ded.exe') -Value 'test-data-only' -NoNewline
Set-Content -LiteralPath (Join-Path $root 'baseq2/game.dll') -Value 'test-data-only' -NoNewline
$receipt=@{version='combat_diagnostic_engine_v1';binaries=@{}}
foreach($name in @('q2ded.exe','baseq2/game.dll')){$receipt.binaries[$name]=(Get-FileHash -LiteralPath (Join-Path $root $name)).Hash.ToLowerInvariant()}
$manifest=Join-Path $root 'engine.json'
$receipt|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $manifest
$valid=Resolve-HarnessDiagnosticEngine $root
if($valid.ServerExe -ne (Join-Path $root 'q2ded.exe')){throw 'Diagnostic path differs'}
$receipt.binaries['q2ded.exe']='0'*64
$receipt|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $manifest
$failed=$false
try{$null=Resolve-HarnessDiagnosticEngine $root}catch{$failed=$true}
if(!$failed){throw 'Changed engine hash accepted'}
$receipt.version='unknown'
$receipt|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $manifest
$failed=$false
try{$null=Resolve-HarnessDiagnosticEngine $root}catch{$failed=$true}
if(!$failed){throw 'Unknown engine version accepted'}
Write-Output "PASS: explicit engine path, hash and version admission ($root)"
