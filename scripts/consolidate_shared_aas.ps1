[CmdletBinding()]
param([switch]$Apply,[string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=(Resolve-Path -LiteralPath (Join-Path $repo 'workspace')).Path
$boundary=$root+[IO.Path]::DirectorySeparatorChar
$active=@(Get-CimInstance Win32_Process|Where-Object {$_.Name -match '^(q2ded|q2coopbot.*|q2ppo-data|q2combat-export|bspc|python.*)\.exe$'})
if($active.Count){throw 'Finish AAS writers and combat runs before migration'}
$maintenance=Join-Path $root 'build/artifact-maintenance'
New-Item -ItemType Directory -Path $maintenance -Force|Out-Null
$tag=Get-Date -Format yyyyMMdd-HHmmss
$planPath=Join-Path $maintenance "aas-plan-$tag.json"
& $Python "$PSScriptRoot/inventory_shared_aas.py" --root $root --out $planPath
if($LASTEXITCODE){throw 'AAS inventory failed'}
$plan=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json
if(!$Apply){return}
. "$PSScriptRoot/prepare_runtime.ps1" -FunctionsOnly
New-Item -ItemType Directory -Path $plan.store -Force|Out-Null
$before=(Get-PSDrive F).Free
$replaced=0
foreach($group in $plan.groups) {
    $canonical=Join-Path $plan.store ($group.sha256+'.aas')
    # Copy one independent canonical file; old hard links can then be released.
    if(!(Test-Path -LiteralPath $canonical)){[IO.File]::Copy($group.source,$canonical)}
    if((Get-RuntimeImmutableHash $canonical) -ne $group.sha256){throw 'Canonical AAS mismatch'}
    foreach($target in $group.paths) {
        $absolute=[IO.Path]::GetFullPath($target)
        if(!$absolute.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'AAS target escaped workspace'}
        for($parent=[IO.DirectoryInfo]::new((Split-Path $absolute -Parent));$parent -and $parent.FullName.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase);$parent=$parent.Parent) {
            if($parent.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'AAS migration refuses junction ancestors'}
        }
        $item=Get-Item -LiteralPath $absolute
        if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'AAS candidate changed to a link'}
        if($item.Length -ne $group.bytes){throw 'AAS candidate size changed'}
        $temporary=Join-Path (Split-Path $absolute -Parent) ('.a'+[guid]::NewGuid().ToString('N')+'.tmp')
        try {
            [IO.File]::CreateSymbolicLink($temporary,$canonical)|Out-Null
            [IO.File]::Move($temporary,$absolute,$true)
        }finally{if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary}}
        $replaced++
    }
    Write-Output "Consolidated $($group.sha256): $($group.paths.Count) paths"
}
$verified=0
foreach($group in $plan.groups) {
    $canonical=Join-Path $plan.store ($group.sha256+'.aas')
    if((Get-RuntimeImmutableHash $canonical) -ne $group.sha256){throw 'AAS final hash mismatch'}
    foreach($target in $group.paths) {
        $item=Get-Item -LiteralPath $target
        if($item.LinkType -ne 'SymbolicLink' -or $item.Target -ne $canonical -or !(Test-Path -LiteralPath $target)){throw 'AAS link verification failed'}
        $verified++
    }
}
& "$env:SystemRoot/System32/compact.exe" /C "/S:$($plan.store)" /EXE:LZX /I /Q '*.aas'
if($LASTEXITCODE){throw 'Shared AAS compression failed'}
$receipt=@{version='shared_aas_v1';plan=$planPath;store=$plan.store;unique_contents=$plan.groups.Count;content_bytes=$plan.content_bytes;replaced=$replaced;verified=$verified;free_before=$before;free_after=(Get-PSDrive F).Free;scope='All migrated paths preserve original content; one canonical file per SHA256, including historical graphs.'}
$receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $maintenance "aas-result-$tag.json") -Encoding utf8NoBOM
$receipt|ConvertTo-Json
