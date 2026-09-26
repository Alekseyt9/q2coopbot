[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$AASPath)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repo ('workspace/artifacts/prepare-runtime-test-' + [guid]::NewGuid().ToString('N'))
$assets = Join-Path $output 'assets'
$runtime = Join-Path $output 'runtime'
$update = Join-Path $output 'update'
New-Item -ItemType Directory -Path "$assets/maps", $update -Force | Out-Null
Set-Content "$assets/pak0.pak" 'fixture'
Set-Content "$output/server.exe" 'fixture'
Set-Content "$output/game.dll" 'fixture'
Set-Content "$assets/maps/base2.aas" 'old asset fixture'
$arguments = @{AssetsRoot=$assets; RuntimeRoot=$runtime; ServerExe="$output/server.exe"; GameDll="$output/game.dll"}
& "$PSScriptRoot/prepare_runtime.ps1" @arguments
$target = "$runtime/baseq2/maps/base2.aas"
$live = "$output/live-base2.aas"
New-Item -ItemType HardLink -Path $live -Target $target | Out-Null
$oldHash = (Get-FileHash $live).Hash
Copy-Item -LiteralPath $AASPath -Destination "$update/base2.aas"
& "$PSScriptRoot/prepare_runtime.ps1" @arguments -AASRoot $update
$expected = (Get-FileHash $AASPath).Hash
if ((Get-FileHash $target).Hash -ne $expected) { throw 'Explicit AAS update failed' }
if ((Get-FileHash $live).Hash -ne $oldHash) { throw 'Live hard link was modified' }
& "$PSScriptRoot/prepare_runtime.ps1" @arguments
if ((Get-FileHash $target).Hash -ne $expected) { throw 'Asset archive downgraded runtime AAS' }
Write-Output "PASS: explicit update, live hard-link isolation, no implicit downgrade ($output)"
