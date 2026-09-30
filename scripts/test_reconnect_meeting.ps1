$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_reconnect_meeting.ps1"
$positive=@(
    foreach($frame in 1..3){[pscustomobject]@{frame=$frame;self=@(300,0,24);health=100;goal='regroup_after_respawn';goal_point=@(0,0,24);teammate=$null}}
    foreach($frame in 4..7){$x=400+50*($frame-4);$player=[math]::Min(600,$x+100);[pscustomobject]@{frame=$frame;self=@($x,0,24);health=100;goal='follow_teammate';goal_point=@($player,0,24);teammate=@($player,0,24)}}
    [pscustomobject]@{frame=8;self=@(550,0,24);health=100;goal='cover_teammate';goal_point=@(600,0,24);teammate=@(600,0,24)}
)
$null=Test-ReconnectMeeting -Rows $positive -DeathPoint @(0,0,24)
$placement=[pscustomobject]@{frame=0;self=@(800,0,24);health=100;goal='wait_for_teammate';teammate=@(820,0,24);arbitration=@{limit_reason='session_barrier'}}
$null=Test-ReconnectMeeting -Rows (@($placement)+$positive) -DeathPoint @(0,0,24)
foreach($case in @('continued-return','wrong-goal','new-death','no-return','late-meeting','stationary-bot','stationary-player','too-far','wrong-floor','lost-contact')){
    $rows=@(($positive | ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    $death=@(0,0,24)
    switch($case){
        'continued-return' {$rows[4].goal='regroup_after_respawn'}
        'wrong-goal' {$rows[4].goal_point=@(0,0,24)}
        'new-death' {$rows[4].health=0}
        'no-return' {$rows[1].goal='wait_for_teammate'}
        'late-meeting' {$death=@(400,0,24)}
        'stationary-bot' {foreach($row in $rows){$row.self=@(550,0,24)}}
        'stationary-player' {foreach($row in $rows | Where-Object teammate){$row.teammate=@(600,0,24);$row.goal_point=@(600,0,24)}}
        'too-far' {$rows[-1].self=@(1000,0,24)}
        'wrong-floor' {$rows[-1].self=@(550,0,124)}
        'lost-contact' {$rows[-1].teammate=$null}
    }
    $rejected=$false
    try{$null=Test-ReconnectMeeting -Rows $rows -DeathPoint $death}catch{$rejected=$true}
    if(!$rejected){throw "Invalid meeting accepted: $case"}
}
Write-Output 'PASS: positive meeting and 10 negative controls'
$visible=@($positive | Where-Object teammate)
$null=Test-ReconnectMeeting -Rows $visible -DeathPoint @(0,0,24) -VisibleAtStart
foreach($case in @('late-contact','old-return')){
    $rows=@(($visible | ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    if($case -eq 'late-contact'){$rows[0].teammate=$null}else{$rows[0].goal='regroup_after_respawn'}
    $rejected=$false
    try{$null=Test-ReconnectMeeting -Rows $rows -DeathPoint @(0,0,24) -VisibleAtStart}catch{$rejected=$true}
    if(!$rejected){throw "Invalid startup meeting accepted: $case"}
}
Write-Output 'PASS: visible startup meeting and 2 negative controls'
