$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_completed_restart.ps1"
$before=@{completed=$true;death=@(194,1940,-167.875);map='base2';generation=7;server='127.0.0.1:32160|test';frame=100}
$after=@{completed=$true;death=@(194,1940,-167.875);map='base2';generation=7;server='127.0.0.1:32160|test';frame=180}
$rows=@(foreach($frame in 101..181){@{frame=$frame;map='base2';spawncount=7;health=100;self=@(828,2232,-231.875);goal='wait_for_teammate';sent_command=@{Forward=0;Side=0;Up=0;Buttons=0}}})
$null=Test-CompletedReturnRestart -Rows $rows -Before $before -After $after -PreviousFrame 100
$cases=@('unfinished-input','resumed-goal','move-command','new-death','visible-player','near-death-point','short-window','duplicate-frame','wrong-server','lost-completed','changed-death','stale-memory')
foreach($case in $cases){
    $testRows=@(($rows|ConvertTo-Json -Depth 8)|ConvertFrom-Json)
    $testBefore=($before|ConvertTo-Json)|ConvertFrom-Json
    $testAfter=($after|ConvertTo-Json)|ConvertFrom-Json
    switch($case){
        'unfinished-input' {$testBefore.completed=$false}
        'resumed-goal' {$testRows[30].goal='regroup_after_respawn'}
        'move-command' {$testRows[30].sent_command.Forward=400}
        'new-death' {$testRows[30].health=0}
        'visible-player' {$testRows[30]|Add-Member -NotePropertyName teammate -NotePropertyValue @(500,2200,-231)}
        'near-death-point' {$testRows[30].self=@(194,1940,-167.875)}
        'short-window' {$testRows=$testRows[0..20]}
        'duplicate-frame' {$testRows[30].frame=$testRows[29].frame}
        'wrong-server' {$testAfter.server='127.0.0.1:32161|other'}
        'lost-completed' {$testAfter.completed=$false}
        'changed-death' {$testAfter.death=@(500,1940,-167.875)}
        'stale-memory' {$testAfter.frame=99}
    }
    $rejected=$false
    try{$null=Test-CompletedReturnRestart -Rows $testRows -Before $testBefore -After $testAfter -PreviousFrame 100}catch{$rejected=$true}
    if(!$rejected){throw "Completed restart accepted invalid case: $case"}
}
Write-Output 'PASS: completed restart and 12 negative controls'
