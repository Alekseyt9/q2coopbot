[CmdletBinding()]
param([int]$BasePort=30520,[string]$OutputRoot='',[ValidateSet('base1','base2','base3')][string]$Map='base2',[string]$SessionPath='',[switch]$RampProbe,[switch]$FirstLipProbe,[switch]$CornerProbe,[switch]$RiseProbe,[switch]$SecondRiseProbe,[switch]$MomentumProbe)
$ErrorActionPreference='Stop'
if($MomentumProbe){$RiseProbe=$true}
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
        $jobs+=Start-ThreadJob -ArgumentList @($script,$port,$dir,$Map,$SessionPath,$RampProbe.IsPresent,$FirstLipProbe.IsPresent,$CornerProbe.IsPresent,$RiseProbe.IsPresent,$SecondRiseProbe.IsPresent,$MomentumProbe.IsPresent) -ScriptBlock {
            param($script,$port,$dir,$map,$session,[bool]$rampProbe,[bool]$firstLipProbe,[bool]$cornerProbe,[bool]$riseProbe,[bool]$secondRiseProbe,[bool]$momentumProbe)
            & $script -Port $port -OutputRoot $dir -Map $map -SessionPath $session -PreparedRuntime -RampProbe:$rampProbe -FirstLipProbe:$firstLipProbe -CornerProbe:$cornerProbe -RiseProbe:$riseProbe -SecondRiseProbe:$secondRiseProbe -MomentumProbe:$momentumProbe
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
    if(@($reports|Where-Object {!$_.route_metrics -or (!$RampProbe -and !$FirstLipProbe -and !$_.route_metrics.elevator_completed -and $Map -eq 'base3') -or (($RampProbe -or $FirstLipProbe) -and ($RampProbe -and !$_.ramp_probe -or $FirstLipProbe -and !$_.first_lip_probe -or $_.route_metrics.jump_missed -ne 0 -or $_.route_metrics.runup_lost_ground -ne 0)) -or ($CornerProbe -and !$_.corner_probe) -or ($RiseProbe -and (!$_.rise_probe -or $_.route_metrics.jump_missed -ne 0 -or $_.route_metrics.runup_lost_ground -ne 0)) -or ($SecondRiseProbe -and (!$_.second_rise_probe -or $_.route_metrics.jump_missed -ne 0 -or $_.route_metrics.runup_lost_ground -ne 0))}).Count){throw 'Return route metrics incomplete'}
    $frames=@($reports|ForEach-Object {[int]$_.route_metrics.elapsed_frames})
    $paths=@($reports|ForEach-Object {[double]$_.route_metrics.horizontal_path})
    $misses=@($reports|ForEach-Object {[int]$_.route_metrics.jump_missed})
    $summary=@{elapsed_frames_min=($frames|Measure-Object -Minimum).Minimum;elapsed_frames_max=($frames|Measure-Object -Maximum).Maximum;horizontal_path_min=($paths|Measure-Object -Minimum).Minimum;horizontal_path_max=($paths|Measure-Object -Maximum).Maximum;jump_missed_total=($misses|Measure-Object -Sum).Sum}
    @{accepted=$true;map=$Map;ramp_probe=[bool]$RampProbe;first_lip_probe=[bool]$FirstLipProbe;corner_probe=[bool]$CornerProbe;rise_probe=[bool]$RiseProbe;second_rise_probe=[bool]$SecondRiseProbe;momentum_probe=[bool]$MomentumProbe;parallelism=2;timescale=2;route_metrics_summary=$summary;reports=$reports}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $root 'report.json')
    Write-Output "PASS: $root"
}finally{
    $jobs|Remove-Job -Force -ErrorAction SilentlyContinue
}
