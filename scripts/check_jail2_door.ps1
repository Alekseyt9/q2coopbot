function Assert-Jail2DoorFollow([object[]]$Rows) {
    $game=@($Rows | Where-Object {$_.map -eq 'jail2' -and $_.frame -ge 100})
    if($game.Count -lt 100){throw 'Insufficient released observations'}
    $first=$game[0]
    if([math]::Abs($first.self[0]-126.625) -gt 1 -or [math]::Abs($first.self[1]+300.5) -gt 1){throw 'Blocking origin not reproduced'}
    foreach($model in @(49,50)) {
        $m=@($first.movers | Where-Object model -EQ $model)
        if($m.Count -ne 1 -or [math]::Abs($m[0].origin[0]) -ne 98){throw 'Original open door displacement missing'}
    }
    if(@($game | Where-Object {$_.health -ne 68 -or !$_.teammate}).Count){throw 'Damage, healing, death or lost player'}
    $bypass=@($game | Where-Object {$_.arbitration.move_source -eq 'door_route_bypass'})
    if(!$bypass.Count){throw 'Checked bypass not exercised'}
    $last=$game[-1]
    $dx=$last.self[0]-$last.teammate[0];$dy=$last.self[1]-$last.teammate[1]
    $distance=[math]::Sqrt($dx*$dx+$dy*$dy)
    if($last.self[1] -le -100 -or $last.goal -ne 'cover_teammate' -or $distance -gt 160 -or $distance -lt 96){throw 'Door passage and resumed spaced accompaniment not proven'}
    return [pscustomobject]@{bypass_frames=$bypass.Count;final_distance=$distance;health=$last.health;final_position=$last.self}
}
