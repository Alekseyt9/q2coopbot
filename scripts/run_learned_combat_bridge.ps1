[CmdletBinding()]
param(
    [ValidateRange(2,4)][int]$Workers=4,
    [ValidateRange(1,20)][int]$EpisodesPerWorker=2,
    [ValidateSet(1,2)][int]$Timescale=2,
    [ValidateRange(20,500)][int]$GameFrames=300,
    [int]$Seed=11800,
    [ValidateRange(1024,65530)][int]$Port=32996,
    [ValidateRange(1024,65535)][int]$BridgePort=33000,
    [ValidateRange(0,1000)][int]$DelayFirstMS=250,
    [switch]$Mixed,
    [switch]$HealthKit,
    [string]$OutputRoot=''
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
if(Get-NetTCPConnection -State Listen -LocalPort $BridgePort -ErrorAction SilentlyContinue){throw 'Bridge port occupied'}
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/learner-bridge-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output directory required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
$env:GOTOOLCHAIN='auto'
$exe=Join-Path $OutputRoot 'q2policy-bridge.exe'
Push-Location $repo
try{go build -o $exe ./cmd/q2policy-bridge;if($LASTEXITCODE){throw 'Bridge build failed'}}finally{Pop-Location}
$probe=Join-Path $PSScriptRoot 'scenarios/combat-control-probe.json'
$probeHash=(Get-FileHash -LiteralPath $probe).Hash
$sources=Get-HarnessSourceRecords $repo
$fingerprint=Get-HarnessFingerprint $sources
$config=Join-Path $OutputRoot 'remote.json'
$policyVersion='diagnostic_probe_bridge_v1'
@{kind='combat_remote_v1';address="127.0.0.1:$BridgePort";timeout_ms=1500;policy_version=$policyVersion;episode='runner';seed=0}|ConvertTo-Json|Set-Content -LiteralPath $config -Encoding utf8NoBOM
$peer=$null
try{
    $arguments=@('--listen',"127.0.0.1:$BridgePort",'--probe',$probe,'--log',(Join-Path $OutputRoot 'bridge.jsonl'),'--policy-version',$policyVersion,'--delay-first-ms',"$DelayFirstMS")
    $peer=Start-Process $exe -ArgumentList @($arguments|ForEach-Object {'"'+$_+'"'}) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bridge.out') -RedirectStandardError (Join-Path $OutputRoot 'bridge.err')
    $deadline=(Get-Date).AddSeconds(5)
    do{
        Start-Sleep -Milliseconds 100
        if($peer.HasExited -or (Get-Date)-gt $deadline){throw 'Bridge startup failed'}
    }while(!(Get-NetTCPConnection -State Listen -LocalPort $BridgePort -OwningProcess $peer.Id -ErrorAction SilentlyContinue))
    $baseline=Join-Path $OutputRoot 'baseline'
    & "$PSScriptRoot/run_learned_combat_baseline.ps1" -Workers $Workers -EpisodesPerWorker $EpisodesPerWorker -Timescale $Timescale -GameFrames $GameFrames -Loadout blaster -CombatMode learned -Synchronous -Mixed:$Mixed -HealthKit:$HealthKit -RewardConfig (Join-Path $PSScriptRoot 'scenarios/combat-reward-v1.json') -ProviderFile $config -Seed $Seed -Port $Port -OutputRoot $baseline
    $r=Get-Content -LiteralPath (Join-Path $baseline 'report.json') -Raw|ConvertFrom-Json
    $messages=@(Get-Content -LiteralPath (Join-Path $OutputRoot 'bridge.jsonl')|ForEach-Object{ConvertFrom-Json $_}|Where-Object request)
    $mismatches=@($messages|Where-Object{
        $_.request.request -ne $_.response.request -or $_.request.session -ne $_.response.session -or
        $_.request.policy_version -ne $_.response.policy_version -or
        ($_.request.observation.identity|ConvertTo-Json -Compress) -ne ($_.response.action.identity|ConvertTo-Json -Compress)
    }).Count
    $sessions=@($messages.request.session|Sort-Object -Unique)
    $expectedCommands=[int](($r.results|Measure-Object provider_controlled_frames -Sum).Sum)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -and $probeHash -eq (Get-FileHash -LiteralPath $probe).Hash
    $episodes=@(foreach($result in $r.results){
        $name="worker-$($result.worker)-seed-$($result.seed)"
        $transactions=@($messages|Where-Object {$_.request.episode -eq $name -and $_.request.seed -eq $result.seed})
        [pscustomobject]@{worker=$result.worker;seed=$result.seed;episode=$result.episode;messages=$transactions.Count;provider_frames=$result.provider_controlled_frames;matched=($transactions.Count -eq $result.provider_controlled_frames);reward_steps=$result.dataset.reward_steps;synchronous=$result.dataset.synchronous_confirmed;max_latency_us=$result.selection_max_us}
    })
    $accepted=$valid -and $r.capture_complete -and $mismatches -eq 0 -and $messages.Count -eq $expectedCommands -and $sessions.Count -eq $Workers*$EpisodesPerWorker -and @($episodes|Where-Object {!$_.matched -or !$_.synchronous}).Count -eq 0
    @{version='combat_bridge_v1';transport_accepted=$accepted;provenance_valid=$valid;diagnostic_peer=$true;trained_weights=$null;workers=$Workers;timescale=$Timescale;delay_first_ms=$DelayFirstMS;messages=$messages.Count;reply_mismatches=$mismatches;sessions=$sessions.Count;episodes=$episodes;probe_sha256=$probeHash;peer_sha256=(Get-FileHash -LiteralPath $exe).Hash;source_fingerprint=$fingerprint;scope='Decision transport only; native execution and offline rewards are verified separately. No online reset/reward API or training acceptance.'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $OutputRoot 'bridge-report.json') -Encoding utf8NoBOM
    "Bridge artifacts: $OutputRoot"
    if(!$accepted){throw 'Bridge transport rejected; inspect preserved reports'}
}finally{
    if($peer -and !$peer.HasExited){Stop-Process -Id $peer.Id}
}
