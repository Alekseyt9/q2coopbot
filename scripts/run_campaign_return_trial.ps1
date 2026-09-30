[CmdletBinding()]
param([int]$BasePort=31950,[string]$OutputRoot='',[switch]$ReloadSameMap,[switch]$ReconnectReturn,[switch]$MeetPlayer,[switch]$CombinedReturn,[switch]$RepeatDeath,[switch]$MapAfterMeeting)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if($MapAfterMeeting){$RepeatDeath=$true}
if($RepeatDeath){$CombinedReturn=$true;$MeetPlayer=$true}
if($ReloadSameMap -and $ReconnectReturn){throw 'Select one boundary mode'}
if($CombinedReturn -and ($ReloadSameMap -or $ReconnectReturn)){throw 'CombinedReturn requires its own session'}
if($MeetPlayer -and !$ReconnectReturn -and !$CombinedReturn){throw 'MeetPlayer requires ReconnectReturn or CombinedReturn'}
. "$PSScriptRoot/check_reconnect_meeting.ps1"
. "$PSScriptRoot/check_repeat_player_recovery.ps1"
. "$PSScriptRoot/check_post_meeting_map.ps1"
if(!$OutputRoot){$prefix=if($ReconnectReturn){'reconnect-return-'}elseif($ReloadSameMap){'same-map-return-'}else{'campaign-return-'};$OutputRoot=Join-Path $repo ('workspace/artifacts/'+$prefix+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path (Get-Location) $OutputRoot}))
if(Test-Path -LiteralPath $root){throw "Output already exists: $root"}
New-Item -ItemType Directory $root|Out-Null
$runtime=Join-Path $root 'source-runtime'
$expectedMaps=if($CombinedReturn -and $MeetPlayer){@('base1','base2','base2','base2','base3','base2','base2')}elseif($CombinedReturn){@('base1','base2','base2','base2','base3')}elseif($ReconnectReturn){@('base2','base2')}elseif($ReloadSameMap){@('base2','base2','base2')}else{@('base1','base2','base3')}
if($MapAfterMeeting){$expectedMaps+=@('base3')}
$session=if($CombinedReturn){'combined-death-return-session.json'}elseif($ReconnectReturn){'reconnect-death-return-session.json'}elseif($ReloadSameMap){'same-map-death-return-session.json'}else{'campaign-death-return-session.json'}
if($MeetPlayer){$session=if($CombinedReturn){'combined-return-meet-player-session.json'}else{'reconnect-return-meet-player-session.json'}}
if($RepeatDeath){$session='combined-meeting-repeat-death-session.json'}
if($MapAfterMeeting){$session='combined-meeting-map-change-session.json'}
foreach($map in @($expectedMaps|Sort-Object -Unique)){& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map $map -RuntimeRoot $runtime|Out-Null}
@{version=1;sessions=@([IO.Path]::GetRelativePath($root,(Join-Path $PSScriptRoot ('scenarios/'+$session))));scenarios=@();timescales=@(2);repetitions=2;parallelism=2;base_port=$BasePort;runtime_root='source-runtime';tail_frames=5}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $root 'suite.json') -Encoding utf8
try{$messages=@(& "$PSScriptRoot/run_scenario_suite.ps1" -Config (Join-Path $root 'suite.json'))}catch{
    @{accepted=$false;stage='native_suite';reason=$_.Exception.Message}|ConvertTo-Json|Set-Content (Join-Path $root 'report.json') -Encoding utf8
    throw
}
$messages|Out-Host
$line=@($messages|Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'})
if($line.Count -ne 1){throw 'Missing suite result'}
$null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$';$suiteRoot=$Matches[1]
$suite=Get-Content (Join-Path $suiteRoot 'suite-summary.json') -Raw|ConvertFrom-Json
try{$results=@(foreach($run in $suite.results){
    if(!$run.accepted -or !$suite.provenance_valid){throw "Campaign session rejected: $($run.directory)"}
    $rows=@(Get-Content (Join-Path $run.directory 'observer.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
    $groups=@($rows|Group-Object connection|Sort-Object {[int]$_.Name})
    if($groups.Count -ne $expectedMaps.Count){throw 'Unexpected number of connection phases'}
    if($ReconnectReturn -and @($rows.spawncount|Sort-Object -Unique).Count -ne 1){throw 'Reconnect changed map generation'}
    $expectedGenerations=if($MapAfterMeeting){6}elseif($CombinedReturn -and $MeetPlayer){5}elseif($CombinedReturn){4}else{$expectedMaps.Count}
    if(!$ReconnectReturn -and @($rows.spawncount|Sort-Object -Unique).Count -ne $expectedGenerations){throw 'Map generation did not change'}
    $processes=Get-Content (Join-Path $run.directory 'session-processes.json') -Raw|ConvertFrom-Json
    if($processes.port -ne $run.port -or !$processes.server_pid -or !$processes.observer_pid){throw 'Missing process identity'}
    $deathPoint=$null
    $phases=@(for($i=0;$i -lt $expectedMaps.Count;$i++){
        $phase=@($groups[$i].Group);$map=$expectedMaps[$i]
        if(@($phase.map|Sort-Object -Unique).Count -ne 1 -or $phase[0].map -ne $map){throw 'Wrong campaign map order'}
        if(@($phase.connection|Sort-Object -Unique).Count -ne 1 -or $phase[0].connection -ne $i+1){throw 'Unexpected connection generation'}
        $dead=@($phase|Where-Object {$_.health -le 0}|Select-Object -First 1)
        $reconnectPhase=($ReconnectReturn -and $i -eq 1) -or ($CombinedReturn -and ($i -eq 2 -or ($MeetPlayer -and $i -eq 6)))
        if($CombinedReturn -and $i -gt 0){
            $previous=@($groups[$i-1].Group)
            if($reconnectPhase){
                if($phase[0].spawncount -ne $previous[-1].spawncount -or $phase[0].frame -le $previous[-1].frame){throw 'Combined reconnect changed generation or rewound frame'}
            }elseif($phase[0].spawncount -eq $previous[-1].spawncount){throw 'Combined map entry retained old generation'}
        }
        if($reconnectPhase){
            if(!($RepeatDeath -and $i -eq 6) -and ($dead.Count -or @($phase|Where-Object {$_.test_observer_kill}).Count)){throw 'Reconnect phase must not create a new death target'}
        }else{
            if(!$dead.Count){throw "No death in $map"}
            $deathPoint=$dead[0].self
            if(@($phase|Where-Object {$_.frame -lt $dead[0].frame -and $_.goal -eq 'regroup_after_respawn'}).Count){throw "Old return leaked into $map"}
        }
        $return=@($phase|Where-Object {$_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0})
        if($RepeatDeath -and $i -eq 6){
            $recovery=Test-RepeatPlayerRecovery -Rows $phase -OldDeath $deathPoint
            $beforeDeath=@($phase|Where-Object {$_.frame -lt $recovery.death_frame})
            $lastVisible=@($beforeDeath|Where-Object teammate|Select-Object -Last 1)
            $meetingRows=@($beforeDeath|Where-Object {$_.frame -le $lastVisible[0].frame})
            $meeting=Test-ReconnectMeeting -Rows $meetingRows -DeathPoint $deathPoint
            $return=@($return|Where-Object {$_.frame -lt $recovery.death_frame})
        }elseif($MeetPlayer -and !$MapAfterMeeting -and $i -eq $expectedMaps.Count-1){$meeting=Test-ReconnectMeeting -Rows $phase -DeathPoint $deathPoint}
        if($return.Count -lt 3){throw "No sustained death-only return in $map"}
        foreach($row in $return){
            if($row.teammate -or $row.last_teammate -or !$row.goal_point){throw 'Return used player memory'}
            foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$deathPoint[$axis]) -gt .125){throw 'Wrong generation death target'}}
        }
        if($MapAfterMeeting -and $i -eq 7){$mapReset=Test-PostMeetingMap -Previous @($groups[6].Group) -Rows $phase}
        if($i -lt $expectedMaps.Count-1 -and !($MapAfterMeeting -and $i -eq 6) -and $phase[-1].goal -ne 'regroup_after_respawn'){throw 'Boundary did not interrupt active return'}
        if($ReconnectReturn -and !$MeetPlayer -and $i -eq 1 -and ($phase[-1].goal -ne 'wait_for_teammate' -or [math]::Sqrt([math]::Pow($phase[-1].self[0]-$deathPoint[0],2)+[math]::Pow($phase[-1].self[1]-$deathPoint[1],2)) -gt 64 -or [math]::Abs($phase[-1].self[2]-$deathPoint[2]) -gt 40)){throw 'Reconnect return did not arrive'}
        @{map=$map;generation=$phase[0].spawncount;death=$deathPoint;return_frames=$return.Count;boundary_goal=$phase[-1].goal}
    })
    $memory=Get-Content (Join-Path $run.directory 'travel-memory.json') -Raw|ConvertFrom-Json
    $cfg=Get-Content (Join-Path $run.directory 'observer-config.json') -Raw|ConvertFrom-Json
    if($memory.map -ne $expectedMaps[-1] -or $memory.generation -ne $phases[-1].generation -or $memory.server -ne "127.0.0.1:$($run.port)|$($cfg.client.memory_session)"){throw 'Persisted memory belongs to wrong server generation'}
    if($MeetPlayer -and !$MapAfterMeeting){
        if($memory.completed -or !$memory.player){throw 'Meeting did not persist player rendezvous'}
        $expectedPlayer=if($RepeatDeath){$recovery.last_player}else{$meeting.last_player}
        foreach($axis in 0..2){if([math]::Abs($memory.player[$axis]-$expectedPlayer[$axis]) -gt .125){throw 'Wrong persisted player position'}}
    }elseif(!$memory.completed -or $memory.player){throw 'Persisted memory did not finish in current server generation'}
    $expectedDeath=if($RepeatDeath -and !$MapAfterMeeting){$recovery.death}else{$phases[-1].death}
    foreach($axis in 0..2){if(!$memory.death -or [math]::Abs($memory.death[$axis]-$expectedDeath[$axis]) -gt .125){throw 'Persisted old death point'}}
    @{accepted=$true;directory=$run.directory;server_pid=$processes.server_pid;observer_pid=$processes.observer_pid;phases=$phases;post_meeting_map=$(if($MapAfterMeeting){$mapReset}else{$null});repeated_death=$(if($RepeatDeath){$recovery}else{$null});meeting=$(if($MeetPlayer){$meeting}else{$null});final_memory_map=$memory.map;final_memory_completed=$memory.completed}
})}catch{
    @{accepted=$false;stage='campaign_validation';reason=$_.Exception.Message;suite=$suiteRoot}|ConvertTo-Json|Set-Content (Join-Path $root 'report.json') -Encoding utf8
    throw
}
if($results.Count -ne 2){throw 'Expected two independent sessions'}
if(@($results.server_pid|Sort-Object -Unique).Count -ne 2 -or @($results.observer_pid|Sort-Object -Unique).Count -ne 2){throw 'Independent processes not verified'}
@{accepted=$true;timescale=2;parallelism=2;map_after_meeting=[bool]$MapAfterMeeting;repeat_death=[bool]$RepeatDeath;combined_return=[bool]$CombinedReturn;reload_same_map=[bool]$ReloadSameMap;reconnect_return=[bool]$ReconnectReturn;meet_player=[bool]$MeetPlayer;scope='forced_connection_boundaries_during_death_return_without_combat';suite=$suiteRoot;results=$results}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $root 'report.json') -Encoding utf8
Write-Output "PASS: $root"
