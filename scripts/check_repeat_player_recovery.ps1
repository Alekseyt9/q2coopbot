function Test-RepeatPlayerRecovery($Rows,$OldDeath) {
    $dead=@($Rows|Where-Object {$_.health -le 0}|Select-Object -First 1)
    if(!$dead.Count){throw 'No repeated death after meeting'}
    $before=@($Rows|Where-Object {$_.frame -lt $dead[0].frame -and $_.teammate})
    if(!$before.Count){throw 'No observed player before repeated death'}
    $player=$before[-1].teammate
    $death=$dead[0].self
    if((Get-MeetingXYDistance $death $OldDeath) -le 128){throw 'Repeated death did not replace old death point'}
    $alive=@($Rows|Where-Object {$_.frame -gt $dead[0].frame -and $_.health -gt 0})
    if(!$alive.Count){throw 'No policy respawn after meeting'}
    $respawn=$alive[0].frame
    if(@($Rows|Where-Object {$_.frame -gt $respawn -and $_.health -le 0}).Count){throw 'Another death during player recovery'}
    $return=@($alive|Where-Object goal -eq 'regroup_after_respawn')
    if($return.Count -lt 3){throw 'No sustained return to last player'}
    foreach($row in $return){
        if($row.teammate -or !$row.last_teammate -or !$row.goal_point){throw 'Unproven hidden player return'}
        foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$player[$axis]) -gt .125 -or [math]::Abs($row.last_teammate[$axis]-$player[$axis]) -gt .125){throw 'Repeated death used stale target'}}
    }
    $arrival=@($alive|Where-Object {$_.frame -ge $return[2].frame -and !$_.teammate -and (Get-MeetingXYDistance $_.self $player) -le 64 -and [math]::Abs($_.self[2]-$player[2]) -le 40}|Select-Object -First 1)
    if(!$arrival.Count){throw 'Bot did not reach remembered player position'}
    $contact=@($alive|Where-Object teammate|Select-Object -First 1)
    if(!$contact.Count -or $contact[0].frame -le $return[2].frame){throw 'No player reacquisition after hidden recovery'}
    $tail=@($Rows|Select-Object -Last 20)
    foreach($row in $tail){if(!$row.teammate -or $row.goal -notin @('follow_teammate','cover_teammate') -or (Get-MeetingXYDistance $row.self $row.teammate) -gt 128 -or [math]::Abs($row.self[2]-$row.teammate[2]) -gt 40){throw 'Player recovery did not settle into following'}}
    @{death_frame=$dead[0].frame;death=$death;last_player_before_death=$player;respawn_frame=$respawn;return_frames=$return.Count;arrival_frame=$arrival[0].frame;contact_frame=$contact[0].frame;last_player=$tail[-1].teammate}
}
