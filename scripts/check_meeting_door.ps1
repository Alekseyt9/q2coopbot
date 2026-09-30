function Test-MeetingDoor($Rows,[int]$ReleaseFrame=140) {
    $rows=@($Rows|Where-Object {$_.frame -ge $ReleaseFrame -and $_.arbitration.limit_reason -ne 'test_setup_hold'})
    if($rows.Count -lt 20){throw 'Missing released gameplay'}
    if(@($rows|Where-Object {$_.health -le 0 -or $_.health -lt $rows[0].health}).Count){throw 'Death or damage after door release'}
    $follow=@($rows|Where-Object {$_.teammate -and $_.goal -eq 'follow_teammate'})
    if($follow.Count -lt 3){throw 'Missing following after door release'}
    foreach($row in $follow){
        if(!$row.goal_point){throw 'Missing observed player goal'}
        foreach($axis in 0..2){if([math]::Abs($row.goal_point[$axis]-$row.teammate[$axis]) -gt .125){throw 'Wrong player goal'}}
    }
    $movement=Get-MeetingXYDistance $rows[0].self $rows[-1].self
    if($movement -lt 128){throw 'Door approach did not advance'}
    foreach($row in @($rows|Select-Object -Last 20)){
        if(!$row.teammate -or $row.goal -notin @('follow_teammate','cover_teammate') -or (Get-MeetingXYDistance $row.self $row.teammate) -gt 128 -or [math]::Abs($row.self[2]-$row.teammate[2]) -gt 40){throw 'Bot did not catch up beyond door'}
    }
    @{movement=$movement;released_health=$rows[0].health;final_health=$rows[-1].health;final_distance=(Get-MeetingXYDistance $rows[-1].self $rows[-1].teammate);short_approach_frames=@($rows|Where-Object {$_.arbitration.move_limit_reason -eq 'door_short_approach'}).Count}
}
