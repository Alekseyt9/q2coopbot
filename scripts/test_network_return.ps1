$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_network_return.ps1"
$rows=@(
    foreach($frame in 1..20){[pscustomobject]@{frame=$frame;map='base2';spawncount=123;self=@(500,0,24);health=100;on_ground=$true;goal='regroup_after_respawn';goal_point=@(0,0,24)}}
    foreach($frame in 51..55){[pscustomobject]@{frame=$frame;map='base2';spawncount=123;self=@(200,0,24);health=100;on_ground=$true;goal='regroup_after_respawn';goal_point=@(0,0,24)}}
    foreach($frame in 56..70){[pscustomobject]@{frame=$frame;map='base2';spawncount=123;self=@(32,0,24);health=100;on_ground=$true;goal='wait_for_teammate'}}
)
$signal=@{map='base2';generation=123;frame=20;phase=0;role='observer'}
$events=@(
    [pscustomobject]@{direction='client_to_server';stage='unarmed';action='forward';elapsed_ms=0}
    foreach($frame in 21..50){[pscustomobject]@{direction='server_to_client';stage='blackout';action='drop';elapsed_ms=50*($frame-21);frames=@(@{frame=$frame});trigger=$(if($frame -eq 21){$signal}else{$null})}}
    [pscustomobject]@{direction='server_to_client';stage='after';action='forward';elapsed_ms=1500;frames=@(@{frame=51})}
    [pscustomobject]@{direction='client_to_server';stage='after';action='forward';elapsed_ms=1501}
)
$memory=@{death=@(0,0,24);completed=$true;map='base2';generation=123}
$null=Test-NetworkReturn -Rows $rows -Events $events -Signal $signal -Memory $memory
foreach($case in @('no-trigger','wrong-trigger','forwarded-loss','short-loss','leaked-frame','wrong-target','new-death','new-generation','unfinished-memory','no-upstream-recovery','no-arrival','no-resumed-return')){
    $r=@(($rows|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    $e=@(($events|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    $m=($memory|ConvertTo-Json -Depth 8)|ConvertFrom-Json
    switch($case){
        'no-trigger' {$e[1].trigger=$null}
        'wrong-trigger' {$e[1].trigger.generation++}
        'forwarded-loss' {$e[3].action='forward'}
        'short-loss' {foreach($v in $e|Where-Object stage -eq blackout){$v.elapsed_ms=0}}
        'leaked-frame' {$r[20].frame=21}
        'wrong-target' {$r[20].goal_point=@(100,0,24)}
        'new-death' {$r[20].health=0}
        'new-generation' {$r[20].spawncount++}
        'unfinished-memory' {$m.completed=$false}
        'no-upstream-recovery' {$e=$e|Where-Object {!($_.stage -eq 'after' -and $_.direction -eq 'client_to_server')}}
        'no-arrival' {$r[-1].self=@(500,0,24)}
        'no-resumed-return' {foreach($v in $r|Where-Object frame -ge 51){$v.goal='wait_for_teammate'}}
    }
    $rejected=$false
    try{$null=Test-NetworkReturn -Rows $r -Events $e -Signal $signal -Memory $m}catch{$rejected=$true}
    if(!$rejected){throw "Invalid network return accepted: $case"}
}
Write-Output 'PASS: network return and 12 negative controls'
