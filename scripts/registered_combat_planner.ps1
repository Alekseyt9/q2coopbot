# Source-bound q2episode metadata CLI. No policy inference or game execution.
. "$PSScriptRoot/harness_manifest.ps1"

function Assert-RegisteredCombatPlanner {
    param([string]$Repository,[object]$Context,[switch]$CheckSources)
    $repo=[IO.Path]::GetFullPath($Repository)
    if(!$Context -or $Context.repository -ne $repo -or $Context.source_fingerprint -notmatch '^[0-9a-f]{64}$'){throw 'Invalid planner context'}
    $key=Get-HarnessFingerprint @([pscustomobject]@{path='sources';sha256=$Context.source_fingerprint},[pscustomobject]@{path='go';sha256=$Context.go_version})
    $bundle=Join-Path $repo ("workspace/build/registered-planners/"+$key)
    if($Context.bundle -ne $bundle -or $Context.executable -ne (Join-Path $bundle 'q2episode.exe')){throw 'Planner context path differs'}
    $receipt=Get-Content -LiteralPath (Join-Path $bundle 'receipt.json') -Raw|ConvertFrom-Json
    if($receipt.version -ne 'registered_combat_planner_v1' -or $receipt.source_fingerprint -ne $Context.source_fingerprint -or $receipt.go_version -ne $Context.go_version -or $receipt.binary_sha256 -ne $Context.binary_sha256){throw 'Planner receipt binding differs'}
    if((Get-FileHash -LiteralPath $Context.executable).Hash.ToLowerInvariant() -ne $receipt.binary_sha256){throw 'Planner executable changed'}
    if($CheckSources -and (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -ne $Context.source_fingerprint){throw 'Planner sources changed'}
}

function New-RegisteredCombatPlanner {
    param([string]$Repository)
    $repo=[IO.Path]::GetFullPath($Repository)
    $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
    $sources=Get-HarnessSourceRecords $repo
    $fingerprint=Get-HarnessFingerprint $sources
    $base=Join-Path $repo 'workspace/build/registered-planners'
    New-Item -ItemType Directory -Path $base -Force|Out-Null
    $goVersion=(& go version) -join "`n"
    if($LASTEXITCODE){throw 'Cannot read Go toolchain version'}
    $key=Get-HarnessFingerprint @([pscustomobject]@{path='sources';sha256=$fingerprint},[pscustomobject]@{path='go';sha256=$goVersion})
    $bundle=Join-Path $base $key
    if(!(Test-Path -LiteralPath $bundle)){
        $pending=Join-Path $base ($key+'.pending-'+[guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $pending|Out-Null
        $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
        $clock=[Diagnostics.Stopwatch]::StartNew()
        Push-Location $repo
        try{
            & go build -buildvcs=false -o (Join-Path $pending 'q2episode.exe') ./cmd/q2episode
            if($LASTEXITCODE){throw 'Planner build failed; staging retained'}
        }finally{Pop-Location}
        if((Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -ne $fingerprint){throw 'Planner sources changed during build'}
        $digest=(Get-FileHash -LiteralPath (Join-Path $pending 'q2episode.exe')).Hash.ToLowerInvariant()
        @{version='registered_combat_planner_v1';source_fingerprint=$fingerprint;sources=$sources;
          binary_sha256=$digest;go_version=$goVersion;build_seconds=$clock.Elapsed.TotalSeconds}|ConvertTo-Json -Depth 8|
            Set-Content -LiteralPath (Join-Path $pending 'receipt.json') -Encoding utf8NoBOM
        $prefix=[IO.Path]::GetFullPath($base).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
        foreach($path in @($pending,$bundle)){
            if(![IO.Path]::GetFullPath($path).StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Planner publication outside cache root'}
        }
        try{[IO.Directory]::Move($pending,$bundle)}catch{
            if(!(Test-Path -LiteralPath $bundle)){throw}
            $winner=Get-Content -LiteralPath (Join-Path $bundle 'receipt.json') -Raw|ConvertFrom-Json
            if($winner.binary_sha256 -ne $digest){throw 'Concurrent planner build differs'}
        }
    }
    $receipt=Get-Content -LiteralPath (Join-Path $bundle 'receipt.json') -Raw|ConvertFrom-Json
    $context=[pscustomobject]@{repository=$repo;source_fingerprint=$fingerprint;bundle=$bundle;
        executable=(Join-Path $bundle 'q2episode.exe');binary_sha256=$receipt.binary_sha256;go_version=$goVersion;catalogs=@{}}
    Assert-RegisteredCombatPlanner $repo $context -CheckSources
    return $context
}

function Get-RegisteredCombatCatalog {
    param([object]$Context,[string]$Registry,[string]$SHA256)
    $path=[IO.Path]::GetFullPath($Registry)
    if((Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant() -ne $SHA256){throw 'Registry index changed'}
    $key=$path+'|'+$SHA256
    if(!$Context.catalogs.ContainsKey($key)){
        $listing=& $Context.executable --registry $path --list
        if($LASTEXITCODE){throw 'Registry validation failed'}
        $catalog=($listing -join "`n")|ConvertFrom-Json
        if($catalog.path -ne $path -or $catalog.sha256 -ne $SHA256){throw 'Catalog registry binding differs'}
        $Context.catalogs[$key]=$catalog
    }
    # Every caller has just run --verify-plan, which freshly loads/validates all
    # descriptors and compares the complete selected task with regenerated data.
    return $Context.catalogs[$key]
}
