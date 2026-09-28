[CmdletBinding()]
param([int]$BasePort=30520,[string]$OutputRoot='',[ValidateSet('base2','base3')][string]$Map='base2',[string]$SessionPath='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/memory-restart-suite-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=if([IO.Path]::IsPathRooted($OutputRoot)){[IO.Path]::GetFullPath($OutputRoot)}else{[IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))}
if(Test-Path $root){throw "Output already exists: $root"}
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map $Map|Out-Null
New-Item -ItemType Directory $root|Out-Null
$script=Join-Path $PSScriptRoot 'run_memory_restart_trial.ps1'
$jobs=@()
try{
    foreach($index in 0..1){
        $port=$BasePort+$index
        $dir=Join-Path $root ('run-{0:D3}' -f $index)
        $jobs+=Start-ThreadJob -ArgumentList $script,$port,$dir,$Map,$SessionPath -ScriptBlock {
            param($script,$port,$dir,$map,$session)
            & $script -Port $port -OutputRoot $dir -Map $map -SessionPath $session -PreparedRuntime
        }
    }
    $jobs|Wait-Job|Out-Null
    foreach($job in $jobs){
        $output=$job|Receive-Job
        $output|Out-Host
        if($job.State -ne 'Completed' -or @($job.ChildJobs|Where-Object {$_.Error.Count -gt 0}).Count){throw "Restart trial failed: $($job.Id)"}
    }
    $reports=@(foreach($index in 0..1){
        $path=Join-Path $root ('run-{0:D3}/report.json' -f $index)
        if(!(Test-Path $path)){throw "Missing report: $path"}
        Get-Content $path -Raw|ConvertFrom-Json
    })
    if(@($reports|Where-Object {!$_.accepted -or $_.first_pid -eq $_.second_pid}).Count -or $reports[0].server_session -eq $reports[1].server_session -or $reports[0].server_pid -eq $reports[1].server_pid){throw 'Independent process/server restart not verified'}
    @{accepted=$true;map=$Map;parallelism=2;timescale=2;reports=$reports}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $root 'report.json')
    Write-Output "PASS: $root"
}finally{
    $jobs|Remove-Job -Force -ErrorAction SilentlyContinue
}
