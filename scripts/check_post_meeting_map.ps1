function Test-PostMeetingMap($Previous,$Rows) {
    $visible=@($Previous|Where-Object teammate)
    if(!$visible.Count){throw 'No observed player before map change'}
    $before=$visible[-1]
    $rows=@($Rows|Where-Object {$_.arbitration.limit_reason -ne 'session_barrier'})
    if(!$rows.Count -or $rows[0].map -eq $before.map -or $rows[0].spawncount -eq $before.spawncount){throw 'Map generation did not change after meeting'}
    foreach($row in $rows){
        if($row.map -ne $rows[0].map -or $row.spawncount -ne $rows[0].spawncount){throw 'Mixed map generations in recovery phase'}
        if($row.teammate -or $row.last_teammate -or $row.goal -in @('follow_teammate','cover_teammate','search_last_seen','probe_last_seen')){throw 'Player memory leaked across map change'}
    }
    $dead=@($rows|Where-Object {$_.health -le 0}|Select-Object -First 1)
    if(!$dead.Count){throw 'No new-map death'}
    $alive=@($rows|Where-Object {$_.frame -gt $dead[0].frame -and $_.health -gt 0}|Select-Object -First 1)
    if(!$alive.Count -or @($rows|Where-Object {$_.frame -gt $alive[0].frame -and $_.health -le 0}).Count){throw 'Respawn missing or another death in recovery'}
    $initial=@($rows|Where-Object {$_.frame -lt $dead[0].frame -and $_.health -gt 0})
    if($initial.Count -lt 3 -or @($initial|Where-Object {$_.goal -ne 'wait_for_teammate' -or $_.goal_point}).Count){throw 'New map retained an old movement goal'}
    $return=@($rows|Where-Object {$_.frame -gt $dead[0].frame -and $_.health -gt 0 -and $_.goal -eq 'regroup_after_respawn'})
    if($return.Count -lt 3){throw 'No return to new-map death'}
    foreach($row in $return){
        if(!$row.goal_point){throw 'Missing new-map death target'}
        foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$dead[0].self[$axis]) -gt .125){throw 'Return target belongs to previous map'}}
    }
    $last=$rows[-1]
    if($last.health -le 0 -or $last.goal -ne 'wait_for_teammate' -or (Get-MeetingXYDistance $last.self $dead[0].self) -gt 64 -or [math]::Abs($last.self[2]-$dead[0].self[2]) -gt 40){throw 'New-map return did not arrive'}
    @{previous_map=$before.map;previous_player=$before.teammate;map=$last.map;generation=$last.spawncount;idle_frames_before_death=$initial.Count;new_death=$dead[0].self;return_frames=$return.Count;final_distance=(Get-MeetingXYDistance $last.self $dead[0].self)}
}
