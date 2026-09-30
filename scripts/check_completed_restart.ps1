function Test-CompletedReturnRestart($Rows,$Before,$After,[int]$PreviousFrame) {
    if(!$Before.completed -or !$Before.death -or $Before.player){throw 'Restart input is not a completed death return'}
    if($Rows.Count -lt 81 -or $Rows[-1].frame-$Rows[0].frame -lt 80 -or $Rows[0].frame -le $PreviousFrame){throw 'Completed restart observation window incomplete'}
    $lastFrame=$PreviousFrame
    foreach($row in $Rows){
        if($row.frame -le $lastFrame){throw 'Completed restart frames are not monotonic'}
        $lastFrame=$row.frame
        if($row.map -ne $Before.map -or $row.spawncount -ne $Before.generation -or $row.health -le 0 -or $row.teammate -or $row.last_teammate -or $row.goal -ne 'wait_for_teammate' -or $row.goal_point -or $row.test_observer_kill){throw 'Completed return resumed or restart changed context'}
        if($row.sent_command.Forward -or $row.sent_command.Side -or $row.sent_command.Up -or $row.sent_command.Buttons){throw 'Completed restart sent gameplay action'}
        if([math]::Sqrt([math]::Pow($row.self[0]-$Before.death[0],2)+[math]::Pow($row.self[1]-$Before.death[1],2)) -le 128){throw 'Restart too close to completed death point'}
    }
    if(!$After.completed -or $After.player -or !$After.death -or $After.server -ne $Before.server -or $After.map -ne $Before.map -or $After.generation -ne $Before.generation -or $After.frame -lt $Rows[0].frame -or $After.frame -gt $Rows[-1].frame){throw 'Completed memory lost after restart'}
    foreach($axis in 0..2){if([math]::Abs($After.death[$axis]-$Before.death[$axis]) -gt .125){throw 'Completed death point changed'}}
    @{accepted=$true;frames=$Rows[-1].frame-$Rows[0].frame;observations=$Rows.Count;first_frame=$Rows[0].frame;last_frame=$Rows[-1].frame;completed=$After.completed;goal=$Rows[-1].goal}
}
