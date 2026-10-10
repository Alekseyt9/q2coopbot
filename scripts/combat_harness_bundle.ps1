# Immutable, source-bound client/exporter binaries shared by independent runs.
function Assert-HarnessBinaryBundle {
    param([string]$Bundle,[string]$Fingerprint)
    $receipt=Get-Content -LiteralPath (Join-Path $Bundle 'receipt.json') -Raw|ConvertFrom-Json
    if($receipt.version -ne 'combat_harness_binary_bundle_v1' -or $receipt.source_fingerprint -ne $Fingerprint){throw 'Harness bundle source binding differs'}
    foreach($name in @('q2coopbot.exe','q2combat-export.exe')){
        $entry=$receipt.binaries.$name
        if(!$entry -or (Get-FileHash -LiteralPath (Join-Path $Bundle $name)).Hash.ToLowerInvariant() -ne $entry.sha256){throw "Harness bundle binary changed: $name"}
    }
    return $receipt
}

function Install-HarnessBinaryLink {
    param([string]$Source,[string]$Destination,[string]$SHA256)
    if(Test-Path -LiteralPath $Destination){throw 'Fresh binary destination required'}
    if([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Source)) -ne [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Destination))){throw 'Harness binary links require the same volume'}
    # Serialize shard creation across pool workers. NTFS permits at most1024
    # links to one inode; a full shard requires one new copy, not one per run.
    $mutex=[Threading.Mutex]::new($false,('Local\q2combat-binary-'+$SHA256.ToLowerInvariant()))
    $locked=$false
    try{
        try{$locked=$mutex.WaitOne(60000)}catch [Threading.AbandonedMutexException]{$locked=$true}
        if(!$locked){throw 'Timed out waiting for binary link shard'}
        $linked=$false
        try{New-Item -ItemType HardLink -Path $Destination -Target $Source -ErrorAction Stop|Out-Null;$linked=$true}catch{}
        if(!$linked){
            $shards=Join-Path (Split-Path $Source -Parent) 'link-shards'
            New-Item -ItemType Directory -Path $shards -Force|Out-Null
            $name=Split-Path $Source -Leaf
            foreach($candidate in @(Get-ChildItem -LiteralPath $shards -Filter ($name+'.*') -File)){
                if((Get-FileHash -LiteralPath $candidate.FullName).Hash.ToLowerInvariant() -ne $SHA256){throw 'Binary link shard changed'}
                try{New-Item -ItemType HardLink -Path $Destination -Target $candidate.FullName -ErrorAction Stop|Out-Null;$linked=$true;break}catch{}
            }
            if(!$linked){
                $candidate=Join-Path $shards ($name+'.'+[guid]::NewGuid().ToString('N'))
                Copy-Item -LiteralPath $Source -Destination $candidate -ErrorAction Stop
                if((Get-FileHash -LiteralPath $candidate).Hash.ToLowerInvariant() -ne $SHA256){throw 'New binary link shard hash differs'}
                New-Item -ItemType HardLink -Path $Destination -Target $candidate -ErrorAction Stop|Out-Null
            }
        }
    }finally{if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
    if((Get-FileHash -LiteralPath $Destination).Hash.ToLowerInvariant() -ne $SHA256){throw 'Installed harness binary hash differs'}
}

function New-HarnessBinaryBundle {
    param([string]$Repository)
    $records=Get-HarnessSourceRecords $Repository
    $fingerprint=Get-HarnessFingerprint $records
    $base=[IO.Path]::GetFullPath((Join-Path $Repository 'workspace/build/harness-bundles'))
    New-Item -ItemType Directory -Path $base -Force|Out-Null
    $bundle=Join-Path $base $fingerprint
    if(Test-Path -LiteralPath $bundle){$null=Assert-HarnessBinaryBundle $bundle $fingerprint;return $bundle}
    $pending=Join-Path $base ($fingerprint+'.pending-'+[guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $pending|Out-Null
    $env:GOCACHE=Join-Path $Repository 'workspace/build/gocache'
    $env:GOTOOLCHAIN='auto'
    $binaries=[ordered]@{}
    $timer=[Diagnostics.Stopwatch]::StartNew()
    Push-Location $Repository
    try{
        foreach($name in @('q2coopbot','q2combat-export')){
            $destination=Join-Path $pending ($name+'.exe')
            go build -buildvcs=false -o $destination "./cmd/$name"
            if($LASTEXITCODE){throw "Bundle build failed: $name; incomplete staging retained"}
            $binaries[$name+'.exe']=@{sha256=(Get-FileHash -LiteralPath $destination).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $destination).Length}
        }
    }finally{Pop-Location}
    if((Get-HarnessFingerprint (Get-HarnessSourceRecords $Repository)) -ne $fingerprint){throw 'Sources changed during bundle build; staging retained'}
    @{version='combat_harness_binary_bundle_v1';source_fingerprint=$fingerprint;binaries=$binaries;sources=$records;build_seconds=$timer.Elapsed.TotalSeconds;go_version=(& go version)}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $pending 'receipt.json') -Encoding utf8NoBOM
    # Both directory move targets are explicitly contained in the bundle root.
    $prefix=$base.TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
    foreach($path in @($pending,$bundle)){
        if(![IO.Path]::GetFullPath($path).StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw 'Bundle move outside intended root'}
    }
    Move-Item -LiteralPath $pending -Destination $bundle -ErrorAction Stop
    $null=Assert-HarnessBinaryBundle $bundle $fingerprint
    return $bundle
}
