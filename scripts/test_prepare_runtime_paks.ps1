$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'prepare_runtime.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Runtime preparer parse failed'}
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
