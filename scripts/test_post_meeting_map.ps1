$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_reconnect_meeting.ps1"
. "$PSScriptRoot/check_post_meeting_map.ps1"
$previous=@([pscustomobject]@{map='base2';spawncount=11;teammate=@(464,2520,24)})
$fixture=@(
    foreach($frame in 40..42){[pscustomobject]@{frame=$frame;map='base3';spawncount=22;health=100;self=@(100,200,24);goal='wait_for_teammate';goal_point=$null;teammate=$null;last_teammate=$null;arbitration=@{}}}
    [pscustomobject]@{frame=43;map='base3';spawncount=22;health=0;self=@(100,200,24);goal='respawn';teammate=$null;last_teammate=$null;arbitration=@{}}
    foreach($frame in 44..46){[pscustomobject]@{frame=$frame;map='base3';spawncount=22;health=100;self=@(800,200,24);goal='regroup_after_respawn';goal_point=@(100,200,24);teammate=$null;last_teammate=$null;arbitration=@{}}}
    [pscustomobject]@{frame=47;map='base3';spawncount=22;health=100;self=@(120,200,24);goal='wait_for_teammate';teammate=$null;last_teammate=$null;arbitration=@{}}
)
$null=Test-PostMeetingMap -Previous $previous -Rows $fixture
foreach($case in @('same-map','same-generation','mixed-generation','old-player','old-search','old-initial-goal','no-death','no-return','old-return-target','extra-death','no-arrival','wrong-floor')){
    $rows=@(($fixture|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    switch($case){
        'same-map' {foreach($row in $rows){$row.map='base2'}}
        'same-generation' {foreach($row in $rows){$row.spawncount=11}}
        'mixed-generation' {$rows[-1].spawncount=11}
        'old-player' {$rows[0].last_teammate=@(464,2520,24)}
        'old-search' {$rows[0].goal='search_last_seen'}
        'old-initial-goal' {$rows[0].goal_point=@(464,2520,24)}
        'no-death' {$rows[3].health=100}
        'no-return' {$rows[4].goal='wait_for_teammate'}
        'old-return-target' {$rows[4].goal_point=@(464,2520,24)}
        'extra-death' {$extra=[pscustomobject]@{frame=46;map='base3';spawncount=22;health=0;self=@(100,200,24);goal='respawn';teammate=$null;last_teammate=$null;arbitration=@{}};$rows=@($rows[0..6])+@($extra)+@($rows[7])}
        'no-arrival' {$rows[-1].self=@(800,200,24)}
        'wrong-floor' {$rows[-1].self=@(120,200,100)}
    }
    $rejected=$false
    try{$null=Test-PostMeetingMap -Previous $previous -Rows $rows}catch{$rejected=$true}
    if(!$rejected){throw "Invalid post-meeting map accepted: $case"}
}
Write-Output 'PASS: post-meeting map reset and 12 negative controls'
