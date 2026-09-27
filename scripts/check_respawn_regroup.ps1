function Assert-RespawnRegroup($Rows) {
    $details=@(foreach($segment in $Rows | Group-Object spawncount) {
        $rowsInMap=@($segment.Group)
        $dead=@($rowsInMap | Where-Object health -LE 0 | Select-Object -First 1)
        if(!$dead.Count){throw 'Missing real bot death'}
        $return=@($rowsInMap | Where-Object {$_.frame -gt $dead[0].frame -and $_.health -gt 0 -and $_.goal -eq 'regroup_after_respawn' -and !$_.teammate})
        if($return.Count -lt 5){throw 'No hidden-player return after respawn'}
        $first=$return[0];$distance=0.0
        foreach($axis in 0..2){$distance+=[math]::Pow($first.self[$axis]-$first.last_teammate[$axis],2)}
        if([math]::Sqrt($distance) -le 640){throw 'Return did not exceed ordinary search radius'}
        $arrived=@($rowsInMap | Where-Object {$_.frame -gt $first.frame -and $_.health -gt 0 -and $_.teammate -and $_.goal -eq 'cover_teammate'} | Select-Object -First 1)
        if(!$arrived.Count -or $arrived[0].frame-$first.frame -gt 300){throw 'No bounded regroup arrival'}
        [pscustomobject]@{generation=$segment.Name;respawn_frame=$first.frame;arrival_frame=$arrived[0].frame;initial_distance=[math]::Sqrt($distance);hidden_return_frames=$return.Count}
    })
    if($details.Count -ne 2){throw 'Need two map generations'}
    $details
}
