[CmdletBinding()]
param(
    [Parameter(Mandatory)][string[]]$Suite,
    [Parameter(Mandatory)][string]$Models,
    [string]$OutputRoot,
    [ValidateRange(1,16)][int]$MaxInstances=12,
    [ValidateRange(1024,65535)][int]$Port=35200,
    [string]$BinaryBundle,
    [switch]$ValidateOnly,
    [switch]$ReportOnly
)

$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$weapons=@('Blaster','Shotgun','Super Shotgun','Machinegun','Chaingun','Grenade Launcher',
    'Rocket Launcher','HyperBlaster','Railgun','BFG10K','Grenades')

function Read-Json([string]$Path){
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
}

function Get-Sha256([string]$Path){
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-Hashes($Hashes){
    foreach($Path in $Hashes.Keys){
        if((Get-Sha256 $Path) -ne $Hashes[$Path]){throw "Input changed: $Path"}
    }
}

$inputHashes=[ordered]@{}
$modelsPath=(Resolve-Path -LiteralPath $Models).Path
$inputHashes[$modelsPath]=Get-Sha256 $modelsPath
$variants=@(Read-Json $modelsPath)
if($variants.Count -lt 2 -or @($variants.name | Sort-Object -Unique).Count -ne $variants.Count){
    throw 'Provide at least two distinct variants'
}
foreach($variant in $variants){
    if($variant.name -notmatch '^[a-z0-9]+(?:-[a-z0-9]+)*$' -or $variant.mode -notin 'learned','rules'){
        throw 'Invalid model manifest variant'
    }
    if($variant.mode -eq 'learned'){
        $variant.model=(Resolve-Path -LiteralPath $variant.model).Path
        if((Get-Sha256 $variant.model) -ne $variant.sha256){throw "Model digest mismatch: $($variant.name)"}
        $inputHashes[$variant.model]=$variant.sha256
    }
}

foreach($variant in $variants){
    if(!$variant.binary_bundle){continue}
    $variant.binary_bundle=(Resolve-Path -LiteralPath $variant.binary_bundle).Path
    $receiptPath=Join-Path $variant.binary_bundle 'receipt.json'
    $digest=Get-Sha256 $receiptPath
    if($variant.binary_receipt_sha256 -ne $digest){throw "Binary receipt digest mismatch: $($variant.name)"}
    $inputHashes[$receiptPath]=$digest
}

$worlds=[ordered]@{}
$scenes=[ordered]@{}
foreach($file in $Suite){
    $suitePath=(Resolve-Path -LiteralPath $file).Path
    $inputHashes[$suitePath]=Get-Sha256 $suitePath
    $schedule=Read-Json $suitePath
    $localIds=[Collections.Generic.HashSet[string]]::new()
    foreach($job in $schedule.jobs){
        $case=$job.args
        $weapon=[string]$job.weapon
        $key="$($job.seed)/$weapon"
        if(!$localIds.Add($key)){continue}
        if($scenes.Contains($key)){throw "Repeated world and weapon across suites: $key"}
        if($weapon -notin $weapons -or $job.seed -ne $case.Seed){throw "Invalid starting variant: $key"}
        $fixture=(Resolve-Path -LiteralPath $case.GeneratedFixtures).Path
        $digest=Get-Sha256 $fixture
        if($digest -ne $schedule.pins[$fixture]){throw "Fixture digest mismatch: $fixture"}
        $instances=@((Read-Json $fixture).instances)
        if($instances.Count -ne 1){throw "A balanced fixture must contain one world: $key"}
        $instance=$instances[0]
        if($instance.engine_seed -ne $job.seed -or $instance.map -ne $job.map -or
            $instance.inventory.initial_weapon -ne $weapon){
            throw "Fixture identity mismatch: $key"
        }
        $core=[ordered]@{}
        foreach($field in $instance.Keys){
            if($field -ne 'inventory'){ $core[$field]=$instance[$field] }
        }
        $base=ConvertTo-Json -InputObject $core -Depth 20 -Compress
        $id=[string]$job.seed
        if($worlds.Contains($id) -and $worlds[$id] -ne $base){throw "World changes with weapon: $id"}
        $worlds[$id]=$base
        $inputHashes[$fixture]=$digest
        $reward=(Resolve-Path -LiteralPath $case.RewardConfig).Path
        $inputHashes[$reward]=Get-Sha256 $reward
        $scenes[$key]=[ordered]@{
            seed=[int]$job.seed; map=$job.map; weapon=$weapon; fixture=$fixture
            frames=[int]$case.GameFrames; skill=[int]$case.Skill; mixed=[bool]$case.Mixed
            loadout=$case.Loadout; reward=$reward; checked_wall_slide=[bool]$case.CheckedWallSlide
        }
    }
}
if(!$worlds.Count -or $scenes.Count -ne $worlds.Count*$weapons.Count){
    throw 'Each world must include all 11 weapons exactly once'
}
foreach($world in $worlds.Keys){
    foreach($weapon in $weapons){
        if(!$scenes.Contains("$world/$weapon")){throw "Missing $weapon on seed $world"}
    }
}
$gameCount=$scenes.Count*$variants.Count
Write-Output "$($worlds.Count) matched worlds, $($weapons.Count) weapons, $($variants.Count) variants, $gameCount games"
if($ValidateOnly){return}

if(!$OutputRoot){throw 'OutputRoot is required for a native run'}
$root=[IO.Path]::GetFullPath($OutputRoot)
$spec=Join-Path $root 'matrix.json'
if($ReportOnly){
    $matrix=Read-Json $spec
    if($matrix.version -ne 'balanced_weapon_native_v1' -or $matrix.jobs.Count -ne $gameCount){
        throw 'Saved matrix does not match the requested evaluation'
    }
    foreach($path in $inputHashes.Keys){
        if($matrix.inputs[$path] -ne $inputHashes[$path]){throw "Saved matrix input differs: $path"}
    }
    $inputHashes=$matrix.inputs
    Assert-Hashes $inputHashes
    $jobs=@($matrix.jobs)
}else{
    if(Test-Path -LiteralPath $root){throw 'Use a fresh output folder'}
    if($Port+$gameCount-1 -gt 65535){throw 'Port range exceeds 65535'}
    $ports=$Port..($Port+$gameCount-1)
    if(@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object LocalPort -in $ports).Count){
        throw 'Evaluation ports are occupied'
    }
    if(!$env:Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT){throw 'Set Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT to the native objective engine'}
    $enginePath=Join-Path $env:Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT 'engine.json'
    if((Read-Json $enginePath).capabilities -notcontains 'combat_objective_v1'){
        throw 'Engine does not support native objectives'
    }
    $inputHashes[(Resolve-Path -LiteralPath $enginePath).Path]=Get-Sha256 $enginePath
    . "$PSScriptRoot/harness_manifest.ps1"
    . "$PSScriptRoot/combat_harness_bundle.ps1"
    if($BinaryBundle){
        $bundle=(Resolve-Path -LiteralPath $BinaryBundle).Path
        $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
        $null=Assert-HarnessBinaryBundle $bundle $fingerprint
    }else{
        $bundle=New-HarnessBinaryBundle $repo
    }
    $experimental=@($variants | Where-Object { $_.binary_bundle })
    if($experimental.Count){
        $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
        foreach($variant in $experimental){
            $null=Assert-HarnessBinaryBundle -Bundle $variant.binary_bundle -Fingerprint $fingerprint
        }
    }
    $receipt=Join-Path $bundle 'receipt.json'
    $inputHashes[$receipt]=Get-Sha256 $receipt
    New-Item -ItemType Directory -Path $root | Out-Null
    $jobs=[Collections.Generic.List[object]]::new()
    foreach($scene in $scenes.Values){
        foreach($variant in $variants){
            $index=$jobs.Count
            $contract=if($variant.contract){$variant.contract}elseif($variant.mode -eq 'learned'){'stock_weapons_fire_v1'}else{'legacy_v1'}
            $argsMap=@{
                OutputRoot=Join-Path $root "case-$index"; GeneratedFixtures=$scene.fixture
                Workers=1; EpisodesPerWorker=1; Timescale=2; GameFrames=$scene.frames; Skill=$scene.skill
                Port=$Port+$index; Seed=$scene.seed; Synchronous=$true; ReleaseGameFrame=100
                CombatMode=$variant.mode; ExecutionContract=$contract; Loadout=$scene.loadout; Mixed=$scene.mixed
                BinaryBundle=$(if($variant.binary_bundle){$variant.binary_bundle}else{$bundle}); RequireNativeObjective=$true; StopOnGoal=$true; StopOnFirstDeath=$true
                RewardConfig=$scene.reward; CheckedWallSlide=($scene.checked_wall_slide -or $contract -ne 'legacy_v1')
            }
            if($variant.mode -eq 'learned'){$argsMap.ProviderFile=$variant.model}
            $jobs.Add([ordered]@{index=$index;scene=$scene;variant=$variant.name;argsMap=$argsMap})
        }
    }
    [ordered]@{version='balanced_weapon_native_v1';inputs=$inputHashes;models=$variants;jobs=$jobs} |
        ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $spec -Encoding utf8NoBOM
}
$results=@($jobs | ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    $job=$_
    $argsMap=$job.argsMap
    try{
        if(!$using:ReportOnly){
            & "$using:repo/scripts/run_learned_combat_baseline.ps1" @argsMap *>&1 |
                Set-Content -LiteralPath "$using:root/run-$($job.index).log"
        }
        $record=Get-Content -LiteralPath "$($argsMap.OutputRoot)/report.json" -Raw | ConvertFrom-Json -AsHashtable
        $episode=$record.results[0]
        if(!$record.capture_complete -or !$record.provenance_valid -or $record.results.Count -ne 1 -or
            !$episode.capture_valid -or !$episode.seed_confirmed -or !$episode.dispatch_valid -or
            !$episode.generated_start.confirmed -or $episode.seed -ne $job.scene.seed -or
            $episode.first_life.map -ne $job.scene.map -or
            $episode.generated_fixture.inventory.initial_weapon -ne $job.scene.weapon){
            throw 'Native capture, seed, or initial weapon failed verification'
        }
        $expected=(Get-Content -LiteralPath $argsMap.GeneratedFixtures -Raw | ConvertFrom-Json -AsHashtable).instances[0]
        if((ConvertTo-Json $expected -Depth 20 -Compress) -cne
            (ConvertTo-Json $episode.generated_fixture -Depth 20 -Compress)){
            throw 'Native fixture differs from the registered input'
        }
        [ordered]@{index=$job.index;variant=$job.variant;seed=$job.scene.seed;map=$job.scene.map
            weapon=$job.scene.weapon;clear=[int]([bool]$episode.goal_stop -and !$episode.death_stop)
            death=[int][bool]$episode.death_stop;self_hp=$episode.first_life.damage.self_health_damage}
    }catch{
        [ordered]@{index=$job.index;variant=$job.variant;error=$_.Exception.Message}
    }
} -ThrottleLimit $MaxInstances)
Assert-Hashes $inputHashes
$errors=@($results | Where-Object { $_.Contains('error') })
$summary=@()
if(!$errors.Count -and $results.Count -eq $gameCount){
    foreach($group in ($results | Group-Object -Property { $_['variant'] }, { $_['weapon'] })){
        if($group.Count -ne $worlds.Count){throw 'Unexpected number of fights per weapon'}
        $first=$group.Group[0]
        $summary+=[ordered]@{variant=$first['variant'];weapon=$first['weapon']
            fights=$group.Count;clear=($group.Group | ForEach-Object {$_['clear']} | Measure-Object -Sum).Sum
            death=($group.Group | ForEach-Object {$_['death']} | Measure-Object -Sum).Sum
            self_hp=($group.Group | ForEach-Object {$_['self_hp']} | Measure-Object -Sum).Sum}
    }
}
[ordered]@{state=$(if($summary.Count -eq $weapons.Count*$variants.Count){'complete'}else{'incomplete'});matrix_sha256=(Get-Sha256 $spec)
    games=$results.Count;errors=$errors;per_weapon=$summary;results=$results} |
    ConvertTo-Json -Depth 16 | Set-Content -LiteralPath "$root/report.json" -Encoding utf8NoBOM
if($summary.Count -ne $weapons.Count*$variants.Count){throw "Incomplete balanced evaluation. Inspect $root/report.json"}
Write-Output "$root/report.json"
