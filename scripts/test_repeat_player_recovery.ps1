$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_reconnect_meeting.ps1"
. "$PSScriptRoot/check_repeat_player_recovery.ps1"
$positive=@(
    [pscustomobject]@{frame=1;health=100;self=@(600,0,24);teammate=@(500,0,24);goal='cover_teammate'}
    [pscustomobject]@{frame=2;health=0;self=@(600,0,24);teammate=$null;goal='respawn'}
    foreach($frame in 3..5){[pscustomobject]@{frame=$frame;health=100;self=@(1000,0,24);teammate=$null;last_teammate=@(500,0,24);goal_point=@(500,0,24);goal='regroup_after_respawn'}}
    [pscustomobject]@{frame=6;health=100;self=@(550,0,24);teammate=$null;goal='wait_for_teammate'}
    foreach($frame in 7..26){[pscustomobject]@{frame=$frame;health=100;self=@(550,0,24);teammate=@(500,0,24);goal='cover_teammate'}}
)
$null=Test-RepeatPlayerRecovery -Rows $positive -OldDeath @(0,0,24)
foreach($case in @('no-death','no-observation','old-death','no-respawn','another-death','no-return','death-target','wrong-memory','visible-return','no-hidden-arrival','wrong-floor','no-reacquisition','lost-contact')){
    $rows=@(($positive|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    switch($case){
        'no-death' {$rows[1].health=100}
        'no-observation' {$rows[0].teammate=$null}
        'old-death' {$rows[1].self=@(0,0,24)}
        'no-respawn' {foreach($row in $rows|Where-Object {$_.frame -ge 3}){$row.health=0}}
        'another-death' {$rows[6].health=0}
        'no-return' {$rows[3].goal='wait_for_teammate'}
        'death-target' {$rows[3].goal_point=@(600,0,24)}
        'wrong-memory' {$rows[3].last_teammate=@(0,0,24)}
        'visible-return' {$rows[3].teammate=@(500,0,24)}
        'no-hidden-arrival' {$rows[5].self=@(1000,0,24)}
        'wrong-floor' {$rows[5].self=@(550,0,124)}
        'no-reacquisition' {foreach($row in $rows|Where-Object {$_.frame -ge 7}){$row.teammate=$null}}
        'lost-contact' {$rows[-1].teammate=$null}
    }
    $rejected=$false
    try{$null=Test-RepeatPlayerRecovery -Rows $rows -OldDeath @(0,0,24)}catch{$rejected=$true}
    if(!$rejected){throw "Invalid repeated recovery accepted: $case"}
}
Write-Output 'PASS: repeated player recovery and 13 negative controls'
