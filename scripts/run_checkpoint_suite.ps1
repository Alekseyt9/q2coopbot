[CmdletBinding()]
param([int]$Port=30300)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$source=& "$PSScriptRoot/prepare_registered_pickup_runtime.ps1" -Profile inventory
$out=Join-Path $repo ('workspace/artifacts/checkpoint-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
New-Item -ItemType Directory $out|Out-Null
$client=Join-Path $out 'q2coopbot.exe';$tool=Join-Path $out 'q2checkpoint.exe'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'};go build -o $tool ./cmd/q2checkpoint;if($LASTEXITCODE){throw 'Checkpoint build failed'}}finally{Pop-Location}
$configs=@()
foreach($i in 0..1){$path=Join-Path $out "trial-$i.json";@{runtime_root=$source;client_exe=$client;checkpoint_exe=$tool;output_root=(Join-Path $out "run-$i");port=($Port+$i);timescale=2}|ConvertTo-Json|Set-Content $path -Encoding utf8;$configs+=$path}
$trialScript=Join-Path $PSScriptRoot 'run_checkpoint_trial.ps1'
$results=@($configs|ForEach-Object -Parallel {
    $err='';try{& $using:trialScript -Config $_|Out-Host}catch{$err=$_.Exception.Message}
    $cfg=Get-Content $_ -Raw|ConvertFrom-Json;$report=Join-Path $cfg.output_root 'report.json'
    if(Test-Path $report){Get-Content $report -Raw|ConvertFrom-Json}else{[pscustomobject]@{accepted=$false;reason=$err;port=$cfg.port}}
} -ThrottleLimit 2)
$valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
@{source_fingerprint=$fingerprint;provenance_valid=$valid;parallelism=2;timescale=2;results=$results}|ConvertTo-Json -Depth 35|Set-Content (Join-Path $out 'report.json') -Encoding utf8
Write-Output "Checkpoint suite: $out"
if(!$valid -or $results.Count -ne 2 -or @($results|Where-Object {!$_.accepted}).Count){throw 'Checkpoint suite rejected'}
