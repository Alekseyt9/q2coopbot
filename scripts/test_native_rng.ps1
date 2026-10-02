[CmdletBinding()]
param([string]$Compiler='F:/src/quake2/buildenv/pkg/buildenv/mingw64/bin/gcc.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$source=Join-Path (Split-Path $repo -Parent) 'yquake2/src/common/shared/rand.c'
$exe=Join-Path $repo 'workspace/build/rng_checkpoint.exe'
New-Item -ItemType Directory (Split-Path $exe -Parent) -Force|Out-Null
$oldPath=$env:PATH
try{
    $env:PATH=(Split-Path $Compiler -Parent)+';'+$oldPath
    & $Compiler (Join-Path $PSScriptRoot 'native/rng_checkpoint.c') $source -o $exe
    if($LASTEXITCODE){throw 'Native RNG test build failed'}
    & $exe
    if($LASTEXITCODE){throw 'Native RNG test failed'}
}finally{$env:PATH=$oldPath}
