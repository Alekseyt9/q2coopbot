[CmdletBinding()]
param([int]$BasePort=32520,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_reconnect_meeting.ps1"
. "$PSScriptRoot/check_meeting_door.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/meeting-door-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path (Get-Location) $OutputRoot}))
if(Test-Path $root){throw "Output already exists: $root"}
New-Item -ItemType Directory $root|Out-Null
$runtime=Join-Path $root 'source-runtime'
& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2 -RuntimeRoot $runtime|Out-Null
@{version=1;scenarios=@([IO.Path]::GetRelativePath($root,(Join-Path $PSScriptRoot 'scenarios/base2-meeting-door-approach.json')));sessions=@();timescales=@(2);repetitions=2;parallelism=2;base_port=$BasePort;runtime_root='source-runtime';tail_frames=5}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root 'suite.json') -Encoding utf8
try{
    $messages=@(& "$PSScriptRoot/run_scenario_suite.ps1" -Config (Join-Path $root 'suite.json'))
}catch{
    @{accepted=$false;stage='native_suite';reason=$_.Exception.Message;timescale=2;parallelism=2}|ConvertTo-Json|Set-Content (Join-Path $root 'report.json') -Encoding utf8
    throw
}
$messages|Out-Host
$line=@($messages|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'})
if($line.Count -ne 1){throw 'Missing suite result'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$suiteRoot=$Matches[1]
$suite=Get-Content (Join-Path $suiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$results=@(foreach($run in $suite.results){
    try{
        if(!$run.accepted -or !$suite.provenance_valid){throw 'Native suite or provenance rejected'}
        $trace=Join-Path $run.directory ('scale-2-port-'+$run.port+'-trace.jsonl')
        $rows=@(Get-Content $trace|ForEach-Object {$_|ConvertFrom-Json})
        $proof=Test-MeetingDoor -Rows $rows
        @{accepted=$true;directory=$run.directory;port=$run.port;proof=$proof}
    }catch{@{accepted=$false;directory=$run.directory;port=$run.port;reason=$_.Exception.Message}}
})
$accepted=$results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0 -and @($results.port|Sort-Object -Unique).Count -eq 2
@{accepted=$accepted;suite=$suiteRoot;timescale=2;parallelism=2;scope='captured_door_approach_then_follow';results=$results}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $root 'report.json') -Encoding utf8
if(!$accepted){throw "Door trial rejected; report saved: $root"}
Write-Output "PASS: $root"
