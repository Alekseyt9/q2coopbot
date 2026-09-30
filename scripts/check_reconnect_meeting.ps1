function Get-MeetingXYDistance($a,$b) {
    [math]::Sqrt([math]::Pow($a[0]-$b[0],2)+[math]::Pow($a[1]-$b[1],2))
}

function Test-ReconnectMeeting($Rows,$DeathPoint,[switch]$VisibleAtStart) {
    # Placement snapshots can contain the actor at a default spawn. Only the
    # released gameplay stream proves a meeting during the restored return.
    $Rows=@($Rows | Where-Object {$_.arbitration.limit_reason -ne 'session_barrier'})
    $visible=@($Rows | Where-Object {$_.teammate})
    if($visible.Count -lt 3){throw 'No sustained player contact after reconnect'}
    $first=$visible[0]
    $before=@($Rows | Where-Object {$_.frame -lt $first.frame -and $_.goal -eq 'regroup_after_respawn'})
    if($VisibleAtStart){
        if(!$Rows[0].teammate -or $first.frame -ne $Rows[0].frame -or @($Rows|Where-Object goal -eq 'regroup_after_respawn').Count){throw 'Visible player did not take priority on first snapshot'}
    }elseif($before.Count -lt 3){throw 'Player appeared before return resumed'}
    if((Get-MeetingXYDistance $first.self $DeathPoint) -le 128){throw 'Meeting happened after death point arrival'}
    $after=@($Rows | Where-Object {$_.frame -ge $first.frame})
    if(@($after | Where-Object {$_.goal -eq 'regroup_after_respawn' -or $_.health -le 0}).Count){throw 'Death return continued after meeting'}
    $following=@($visible | Where-Object {$_.goal -eq 'follow_teammate' -and $_.goal_point})
    if($following.Count -lt 3){throw 'Player contact did not resume following'}
    foreach($row in @($visible | Where-Object {$_.goal -in @('follow_teammate','cover_teammate')})){
        if(!$row.goal_point){throw 'Missing player goal'}
        foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$row.teammate[$axis]) -gt .125){throw 'Follow goal is not observed player'}}
    }
    $last=$visible[-1]
    if($Rows[-1].goal -notin @('follow_teammate','cover_teammate') -or !$Rows[-1].teammate){throw 'Player contact lost before completion'}
    if((Get-MeetingXYDistance $first.teammate $last.teammate) -lt 64){throw 'Test player did not lead the bot'}
    if((Get-MeetingXYDistance $first.self $last.self) -lt 64){throw 'Following bot did not move'}
    if((Get-MeetingXYDistance $last.self $last.teammate) -gt 128 -or [math]::Abs($last.self[2]-$last.teammate[2]) -gt 40){throw 'Following bot did not catch up'}
    @{contact_frame=$first.frame;follow_frames=$following.Count;movement=(Get-MeetingXYDistance $first.self $last.self);player_movement=(Get-MeetingXYDistance $first.teammate $last.teammate);final_distance=(Get-MeetingXYDistance $last.self $last.teammate);last_player=$last.teammate}
}
