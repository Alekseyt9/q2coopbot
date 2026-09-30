$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_reconnect_meeting.ps1"
. "$PSScriptRoot/check_meeting_door.ps1"
$fixture=@(
    foreach($frame in 100..102){[pscustomobject]@{frame=$frame;self=@(700,2200,24);health=100;goal='follow_teammate';goal_point=@(500,2500,24);teammate=@(500,2500,24);arbitration=@{}}}
    foreach($frame in 103..122){[pscustomobject]@{frame=$frame;self=@(550,2500,24);health=100;goal='cover_teammate';teammate=@(500,2500,24);arbitration=@{}}}
)
$null=Test-MeetingDoor -Rows $fixture -ReleaseFrame 100
foreach($case in @('short-trace','death','damage','no-follow','wrong-goal','stationary','far','wrong-floor','lost-contact')){
    $rows=@(($fixture|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    switch($case){
        'short-trace' {$rows=$rows[0..2]}
        'death' {$rows[-1].health=0}
        'damage' {$rows[-1].health=99}
        'no-follow' {$rows[1].goal='wait_for_teammate'}
        'wrong-goal' {$rows[1].goal_point=@(0,0,24)}
        'stationary' {foreach($row in $rows){$row.self=@(550,2500,24)}}
        'far' {$rows[-1].self=@(1000,2500,24)}
        'wrong-floor' {$rows[-1].self=@(550,2500,100)}
        'lost-contact' {$rows[-1].teammate=$null}
    }
    $rejected=$false
    try{$null=Test-MeetingDoor -Rows $rows -ReleaseFrame 100}catch{$rejected=$true}
    if(!$rejected){throw "Invalid door follow accepted: $case"}
}
Write-Output 'PASS: door follow and 9 negative controls'
