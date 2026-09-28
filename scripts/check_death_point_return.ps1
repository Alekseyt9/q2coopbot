function Assert-DeathPointReturn($Rows) {
    $groups=@($Rows | Group-Object spawncount | Sort-Object { [int]$_.Group[0].connection },{ [int]$_.Group[0].episode_frame })
    if($groups.Count -ne 2){throw 'Expected two map generations'}
    $results=@(foreach($group in $groups){
        $rows=@($group.Group)
        $dead=@($rows|Where-Object {$_.health -le 0}|Select-Object -First 1)
        if(!$dead.Count){throw 'No real death'}
        $point=$dead[0].self
        $return=@($rows|Where-Object {$_.frame -gt $dead[0].frame -and $_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0})
        if($return.Count -lt 5){throw 'No sustained return'}
        foreach($row in $return){
            if($row.teammate -or $row.last_teammate){throw 'Return used player memory, not death fallback'}
            if(!$row.goal_point){throw 'Missing return target'}
            foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$point[$axis]) -gt 0.125){throw 'Wrong death target'}}
        }
        if(@($return | Where-Object {$_.arbitration.skill -eq 'gap_jump'}).Count){throw 'Unexpected gap jump on walking return route'}
        $first=$return[0]
        $distance=[math]::Sqrt([math]::Pow($first.self[0]-$point[0],2)+[math]::Pow($first.self[1]-$point[1],2))
        if($distance -le 640){throw 'Return did not exceed search limit'}
        $finish=@($rows|Where-Object {$_.frame -gt $first.frame -and $_.health -gt 0 -and $_.goal -ne 'regroup_after_respawn'}|Select-Object -First 1)
        if(!$finish.Count -or $finish[0].frame-$first.frame -gt 300){throw 'Return did not finish within budget'}
        $f=$finish[0]
        $remaining=[math]::Sqrt([math]::Pow($f.self[0]-$point[0],2)+[math]::Pow($f.self[1]-$point[1],2))
        if(!$f.teammate){
            if($f.goal -ne 'wait_for_teammate' -or $remaining -gt 64 -or [math]::Abs($f.self[2]-$point[2]) -gt 40){throw 'Stopped before reaching death point'}
            $case='arrival_without_player'
        }else{
            if($remaining -le 128 -or $f.goal -notin @('follow_teammate','cover_teammate')){throw 'Meeting did not interrupt return early'}
            $case='join_player_en_route'
        }
        if($rows[-1].goal -ne 'cover_teammate'){throw 'No final accompaniment'}
        [pscustomobject]@{case=$case;respawn=$first.frame;finished=$f.frame;initial_distance=$distance;remaining_distance=$remaining;hidden_frames=$return.Count}
    })
    if(@($results.case|Sort-Object -Unique).Count -ne 2){throw 'Both arrival and early meeting cases required'}
    $results
}
