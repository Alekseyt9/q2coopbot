[CmdletBinding()]
param([int]$BasePort=31950,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/campaign-return-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path (Get-Location) $OutputRoot}))
if(Test-Path -LiteralPath $root){throw "Output already exists: $root"}
New-Item -ItemType Directory $root|Out-Null
$runtime=Join-Path $root 'source-runtime'
foreach($map in @('base1','base2','base3')){& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map $map -RuntimeRoot $runtime|Out-Null}
@{version=1;sessions=@([IO.Path]::GetRelativePath($root,(Join-Path $PSScriptRoot 'scenarios/campaign-death-return-session.json')));scenarios=@();timescales=@(2);repetitions=2;parallelism=2;base_port=$BasePort;runtime_root='source-runtime';tail_frames=5}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $root 'suite.json') -Encoding utf8
$messages=@(& "$PSScriptRoot/run_scenario_suite.ps1" -Config (Join-Path $root 'suite.json'))
$messages|Out-Host
$line=@($messages|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'})
if($line.Count -ne 1){throw 'Missing suite result'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$suiteRoot=$Matches[1]
$suite=Get-Content (Join-Path $suiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
$results=@(foreach($run in $suite.results){
    if(!$run.accepted -or !$suite.provenance_valid){throw "Campaign session rejected: $($run.directory)"}
    $rows=@(Get-Content (Join-Path $run.directory 'observer.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
    $groups=@($rows|Group-Object spawncount)
    if($groups.Count -ne 3 -or @($rows.connection|Sort-Object -Unique).Count -ne 1){throw 'Expected three generations in one connection'}
    $phases=@(for($i=0;$i -lt 3;$i++){
        $phase=@($groups[$i].Group);$map=@('base1','base2','base3')[$i]
        if(@($phase.map|Sort-Object -Unique).Count -ne 1 -or $phase[0].map -ne $map){throw 'Wrong campaign map order'}
        $dead=@($phase|Where-Object {$_.health -le 0}|Select-Object -First 1)
        if(!$dead.Count){throw "No death in $map"}
        if(@($phase|Where-Object {$_.frame -lt $dead[0].frame -and $_.goal -eq 'regroup_after_respawn'}).Count){throw "Old return leaked into $map"}
        $return=@($phase|Where-Object {$_.frame -gt $dead[0].frame -and $_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0})
        if($return.Count -lt 3){throw "No sustained death-only return in $map"}
        foreach($row in $return){
            if($row.teammate -or $row.last_teammate -or !$row.goal_point){throw 'Return used player memory'}
            foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$dead[0].self[$axis]) -gt .125){throw 'Wrong generation death target'}}
        }
        if($i -lt 2 -and $phase[-1].goal -ne 'regroup_after_respawn'){throw 'Map transition did not interrupt active return'}
        @{map=$map;generation=$phase[0].spawncount;death=$dead[0].self;return_frames=$return.Count;boundary_goal=$phase[-1].goal}
    })
    $memory=Get-Content (Join-Path $run.directory 'travel-memory.json') -Raw|ConvertFrom-Json
    $cfg=Get-Content (Join-Path $run.directory 'observer-config.json') -Raw|ConvertFrom-Json
    if($memory.map -ne 'base3' -or $memory.generation -ne $phases[-1].generation -or !$memory.completed -or $memory.player -or $memory.server -ne "127.0.0.1:$($run.port)|$($cfg.client.memory_session)"){throw 'Persisted memory did not finish in current server generation'}
    foreach($axis in 0..2){if(!$memory.death -or [math]::Abs($memory.death[$axis]-$phases[-1].death[$axis]) -gt .125){throw 'Persisted old death point'}}
    @{accepted=$true;directory=$run.directory;phases=$phases;final_memory_map=$memory.map;final_memory_completed=$memory.completed}
})
if($results.Count -ne 2){throw 'Expected two independent sessions'}
@{accepted=$true;timescale=2;parallelism=2;scope='forced_map_transitions_during_death_return_without_combat';suite=$suiteRoot;results=$results}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $root 'report.json') -Encoding utf8
Write-Output "PASS: $root"
