$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'prepare_runtime.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Runtime preparer parse failed'}
$hashDefinition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-RuntimeImmutableHash'},$true)
. ([scriptblock]::Create($hashDefinition.Extent.Text))
$definition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Install-RuntimePak'},$true)
. ([scriptblock]::Create($definition.Extent.Text))
$testRoot=Join-Path $repo ('workspace/build/pak-pool-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot|Out-Null
$source=Join-Path $testRoot 'source.pak';[IO.File]::WriteAllText($source,'immutable test archive')
$pool=Join-Path $testRoot 'pool';$target=Join-Path $testRoot 'direct.pak'
Install-RuntimePak $source $target $pool
if((Get-FileHash $source).Hash -ne (Get-FileHash $target).Hash){throw 'Direct link bytes changed'}
$script:blocked=@($source)
function New-Item {
    [CmdletBinding()]param([string]$ItemType,[string]$Path,[string]$Target,[switch]$Force)
    if($ItemType -eq 'HardLink' -and $Target -in $script:blocked){throw 'Injected NTFS link saturation'}
    $args=@{ItemType=$ItemType;Path=$Path;Force=$Force}
    if($Target){$args.Target=$Target}
    Microsoft.PowerShell.Management\New-Item @args
}
$target=Join-Path $testRoot 'pooled.pak'
Install-RuntimePak $source $target $pool
$digest=(Get-FileHash $source).Hash.ToLowerInvariant();$folder=Join-Path $pool $digest
$anchor0=Join-Path $folder 'asset-0.pak'
if(!(Test-Path $anchor0) -or (Get-FileHash $target).Hash -ne (Get-FileHash $source).Hash){throw 'Pool creation failed'}
$script:blocked+=@($anchor0)
Install-RuntimePak $source (Join-Path $testRoot 'rollover.pak') $pool
if(!(Test-Path (Join-Path $folder 'asset-1.pak'))){throw 'Saturated shard did not roll over'}
$corruptPool=Join-Path $testRoot 'corrupt-pool';$corruptFolder=Join-Path $corruptPool $digest
New-Item -ItemType Directory -Path $corruptFolder -Force|Out-Null
[IO.File]::WriteAllText((Join-Path $corruptFolder 'asset-0.pak'),'different content')
$corruptTarget=Join-Path $testRoot 'must-not-exist.pak'
try{Install-RuntimePak $source $corruptTarget $corruptPool;throw 'Corrupt cache accepted'}catch{if($_.Exception.Message -notmatch 'Immutable PAK pool differs'){throw}}
if(Test-Path $corruptTarget){throw 'Corrupt cache created runtime asset'}
if(@(Get-ChildItem $testRoot -Recurse -Filter '*.tmp').Count){throw 'Pool temporary copy leaked'}
'PASS: direct link, saturated source, saturated shard rollover, corrupt pool rejection, temporary cleanup'
Remove-Item Function:New-Item
$definition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Install-RuntimeImmutable'},$true)
. ([scriptblock]::Create($definition.Extent.Text))
$mutable=Join-Path $testRoot 'build-output.bin'
[IO.File]::WriteAllText($mutable,'build version '+[guid]::NewGuid().ToString('N'))
$oldBytes=[IO.File]::ReadAllText($mutable)
$first=Join-Path $testRoot 'runtime-one.bin';$second=Join-Path $testRoot 'runtime-two.bin'
$immutablePool=Join-Path $testRoot 'immutable-pool'
Install-RuntimeImmutable $mutable $first $immutablePool
Install-RuntimeImmutable $mutable $second $immutablePool
if(@(fsutil hardlink list $first).Count -lt 3){throw 'Immutable snapshots are not shared'}
[IO.File]::WriteAllText($mutable,'rebuilt '+[guid]::NewGuid().ToString('N'))
if([IO.File]::ReadAllText($first) -ne $oldBytes){throw 'Source rebuild changed captured runtime'}
Install-RuntimeImmutable $mutable $first $immutablePool
if([IO.File]::ReadAllText($second) -ne $oldBytes){throw 'Runtime replacement modified another runtime'}
if([IO.File]::ReadAllText($first) -ne [IO.File]::ReadAllText($mutable)){throw 'Runtime replacement did not install new bytes'}
'PASS: shared immutable snapshots, source rebuild isolation, atomic runtime replacement'
