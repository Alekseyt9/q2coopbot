[CmdletBinding()]
param([int]$BasePort=32310,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
$records=@(Get-HarnessSourceRecords -Repository $repo)
$fingerprint=Get-HarnessFingerprint -Records $records
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/network-return-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path $root){throw 'Output already exists'}
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2|Out-Null
New-Item -ItemType Directory $root|Out-Null
$jobs=@()
try{
    foreach($index in 0..1){
        $jobs+=Start-ThreadJob -ArgumentList @("$PSScriptRoot/run_network_return_trial.ps1",($BasePort+$index),($BasePort+10+$index),(Join-Path $root ('run-{0:D3}' -f $index))) -ScriptBlock {
            param($script,$port,$proxy,$out)
            & $script -Port $port -ProxyPort $proxy -OutputRoot $out -PreparedRuntime
        }
    }
    $jobs|Wait-Job|Out-Null
    foreach($job in $jobs){$job|Receive-Job|Out-Host;if($job.State -ne 'Completed'){throw 'Network return job failed'}}
    $reports=@(foreach($index in 0..1){Get-Content (Join-Path $root ('run-{0:D3}/report.json' -f $index)) -Raw|ConvertFrom-Json})
    if(@($reports|Where-Object {!$_.accepted}).Count -or $reports[0].server_pid -eq $reports[1].server_pid -or $reports[0].bot_pid -eq $reports[1].bot_pid -or $reports[0].server_session -eq $reports[1].server_session){throw 'Independent native loss trials not proven'}
    if((Get-HarnessFingerprint -Records @(Get-HarnessSourceRecords -Repository $repo)) -ne $fingerprint){throw 'Sources changed during suite'}
    @{source_fingerprint=$fingerprint;source_files=$records;clients=@(foreach($index in 0..1){@{run=$index;sha256=(Get-FileHash (Join-Path $root ('run-{0:D3}/q2coopbot.exe' -f $index))).Hash}})}|ConvertTo-Json -Depth 8|Set-Content "$root/manifest.json"
    @{accepted=$true;provenance_valid=$true;timescale=2;parallelism=2;blackout_ms=1500;reports=$reports}|ConvertTo-Json -Depth 10|Set-Content "$root/report.json"
    Write-Output "PASS: $root"
}finally{$jobs|Remove-Job -Force -ErrorAction SilentlyContinue}
