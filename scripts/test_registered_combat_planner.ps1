[CmdletBinding()]
param([Parameter(Mandatory)][string]$Protocol,[Parameter(Mandatory)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$out=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $out){throw 'Fresh planner test output required'}
New-Item -ItemType Directory -Path $out|Out-Null
. "$PSScriptRoot/registered_combat_planner.ps1"
$protocolPath=(Resolve-Path -LiteralPath $Protocol).Path
$spec=Get-Content -LiteralPath $protocolPath -Raw|ConvertFrom-Json
$plans=@($spec.contexts|ForEach-Object {Join-Path (Split-Path $protocolPath) ($_.id+'-forkC-baseline-plan.json')})
if($plans.Count -ne 8){throw 'Expected eight prior contexts'}
$buildClock=[Diagnostics.Stopwatch]::StartNew()
$context=New-RegisteredCombatPlanner $repo
$buildSeconds=$buildClock.Elapsed.TotalSeconds
$results=@()
function Invoke-PlannerCheck {
    param([string]$Path,[bool]$Legacy)
    $timer=[Diagnostics.Stopwatch]::StartNew();$message='';$accepted=$true
    try{
        $arguments=@{Plan=$Path;DryRun=$true}
        if($Legacy){$arguments.LegacyPlanner=$true}else{$arguments.PlannerContext=$context}
        $message=(& "$PSScriptRoot/run_registered_combat_episodes.ps1" @arguments 2>&1|Out-String).Trim()
    }catch{$accepted=$false;$message=$_.Exception.Message}
    return @{accepted=$accepted;message=$message;seconds=$timer.Elapsed.TotalSeconds}
}
foreach($path in $plans){
    $old=Invoke-PlannerCheck $path $true;$new=Invoke-PlannerCheck $path $false
    if(!$old.accepted -or !$new.accepted -or $old.message -ne $new.message){throw 'Valid-plan parity failed'}
    $results+=@{kind='valid';plan=$path;sha256=(Get-FileHash $path).Hash.ToLowerInvariant();legacy=$old;compiled=$new}
}
$mutations=@('registry_sha','model_sha','runner_sha','seed','episode_sha')
foreach($kind in $mutations){
    $plan=Get-Content -LiteralPath $plans[0] -Raw|ConvertFrom-Json
    switch($kind){
        'registry_sha'{$plan.registry_sha256='0'*64}
        'model_sha'{$plan.model_sha256='0'*64}
        'runner_sha'{$plan.tasks[0].runner_sha256='0'*64}
        'seed'{$plan.tasks[0].seeds=@(0)}
        'episode_sha'{$plan.tasks[0].episode_sha256='0'*64}
    }
    $path=Join-Path $out ($kind+'.json');$plan|ConvertTo-Json -Depth 70|Set-Content -LiteralPath $path -Encoding utf8NoBOM
    $old=Invoke-PlannerCheck $path $true;$new=Invoke-PlannerCheck $path $false
    if($old.accepted -or $new.accepted){throw "Invalid-plan parity failed: $kind"}
    $results+=@{kind=$kind;plan=$path;legacy=$old;compiled=$new}
}
# A descriptor can change without changing index.json: a warm catalog must not
# hide it from the fresh Go VerifyPlan load and complete task comparison.
$privateRegistry=Join-Path $out 'private-registry'
New-Item -ItemType Directory -Path $privateRegistry|Out-Null
$privatePlan=Get-Content -LiteralPath $plans[0] -Raw|ConvertFrom-Json
$descriptorName=$privatePlan.tasks[0].episode.id+'.json'
$descriptor=Join-Path $privateRegistry $descriptorName
Copy-Item -LiteralPath (Join-Path (Split-Path $privatePlan.registry_path) $descriptorName) -Destination $descriptor
$privateIndex=Join-Path $privateRegistry 'index.json'
@{version=1;files=@($descriptorName)}|ConvertTo-Json|Set-Content -LiteralPath $privateIndex -Encoding utf8NoBOM
$privatePlan.registry_path=$privateIndex;$privatePlan.registry_sha256=(Get-FileHash $privateIndex).Hash.ToLowerInvariant()
$privatePath=Join-Path $out 'private-plan.json'
$privatePlan|ConvertTo-Json -Depth 70|Set-Content -LiteralPath $privatePath -Encoding utf8NoBOM
$warm=Invoke-PlannerCheck $privatePath $false
if(!$warm.accepted){throw 'Private registry warmup failed'}
$changed=Get-Content -LiteralPath $descriptor -Raw|ConvertFrom-Json
$changed.title+=' changed';$changed|ConvertTo-Json -Depth 70|Set-Content -LiteralPath $descriptor -Encoding utf8NoBOM
$old=Invoke-PlannerCheck $privatePath $true;$new=Invoke-PlannerCheck $privatePath $false
if($old.accepted -or $new.accepted){throw 'Warm catalog hid a changed descriptor'}
$results+=@{kind='warm_descriptor_drift';legacy=$old;compiled=$new}
# Test damaged binary and source drift in a private fixture, never the live cache.
$fixture=Join-Path $out 'fixture'
foreach($name in @('cmd','internal','scripts')){New-Item -ItemType Directory -Path (Join-Path $fixture $name) -Force|Out-Null}
Set-Content -LiteralPath (Join-Path $fixture 'cmd/fixture.go') -Value 'package main' -Encoding utf8NoBOM
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $fixture)
$key=Get-HarnessFingerprint @([pscustomobject]@{path='sources';sha256=$fingerprint},[pscustomobject]@{path='go';sha256=$context.go_version})
$bundle=Join-Path $fixture ("workspace/build/registered-planners/"+$key)
New-Item -ItemType Directory -Path $bundle -Force|Out-Null
$exe=Join-Path $bundle 'q2episode.exe';Copy-Item -LiteralPath $context.executable -Destination $exe
@{version='registered_combat_planner_v1';source_fingerprint=$fingerprint;go_version=$context.go_version;binary_sha256=$context.binary_sha256}|
    ConvertTo-Json|Set-Content -LiteralPath (Join-Path $bundle 'receipt.json') -Encoding utf8NoBOM
$fake=[pscustomobject]@{repository=$fixture;source_fingerprint=$fingerprint;bundle=$bundle;executable=$exe;binary_sha256=$context.binary_sha256;go_version=$context.go_version}
Assert-RegisteredCombatPlanner $fixture $fake -CheckSources
$file=[IO.File]::Open($exe,[IO.FileMode]::Append);try{$file.WriteByte(0)}finally{$file.Dispose()}
$binaryRejected=$false;try{Assert-RegisteredCombatPlanner $fixture $fake}catch{$binaryRejected=$true}
if(!$binaryRejected){throw 'Changed binary accepted'}
Copy-Item -LiteralPath $context.executable -Destination $exe -Force
Set-Content -LiteralPath (Join-Path $fixture 'cmd/fixture.go') -Value 'package changed' -Encoding utf8NoBOM
$sourceRejected=$false;try{Assert-RegisteredCombatPlanner $fixture $fake -CheckSources}catch{$sourceRejected=$true}
if(!$sourceRejected){throw 'Changed sources accepted'}
Assert-RegisteredCombatPlanner $repo $context -CheckSources
$valid=@($results|Where-Object kind -eq 'valid')
$legacySeconds=($valid|ForEach-Object {$_.legacy.seconds}|Measure-Object -Sum).Sum
$compiledSeconds=($valid|ForEach-Object {$_.compiled.seconds}|Measure-Object -Sum).Sum
@{version='registered_combat_planner_test_v1';state='complete';cases=$results;build_seconds=$buildSeconds;
  legacy_seconds=$legacySeconds;compiled_seconds=$compiledSeconds;speedup=$legacySeconds/$compiledSeconds;
  damaged_binary_rejected=$binaryRejected;source_drift_rejected=$sourceRejected;
  context=($context|Select-Object repository,source_fingerprint,bundle,executable,binary_sha256,go_version);
  scope='Metadata-only CLI acceptance/rejection parity. No Go neural forward, game or CUDA fitting.'}|
    ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $out 'report.json') -Encoding utf8NoBOM
@{state='complete';cases=$results.Count;legacy_seconds=$legacySeconds;compiled_seconds=$compiledSeconds;speedup=$legacySeconds/$compiledSeconds}|ConvertTo-Json -Compress
