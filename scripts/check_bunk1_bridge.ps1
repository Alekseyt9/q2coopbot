function Assert-Bunk1BridgeFollow([object[]]$Rows) {
    $placed=@($Rows | Where-Object {$_.map -eq 'bunk1' -and [math]::Abs($_.self[0]+962.25) -lt 0.5 -and [math]::Abs($_.self[1]+64) -lt 0.5 -and [math]::Abs($_.self[2]-30.125) -lt 0.5})
    if(!$placed.Count){throw 'Captured bottom-lift origin not reproduced'}
    $first=$placed[-1]
    $game=@($Rows | Where-Object {$_.map -eq 'bunk1' -and $_.frame -ge $first.frame})
    if($game.Count -lt 100){throw 'Insufficient released bridge observations'}
    if(@($game | Where-Object {$_.health -ne 87 -or !$_.teammate}).Count){throw 'Damage, healing, death or lost player'}
    if(@($game | Where-Object {$_.arbitration.move_source -eq 'elevator' -or $_.elevator}).Count){throw 'Wrong elevator task retained'}
    $crossing=@($game | Where-Object {$_.arbitration.move_source -eq 'bridge_link' -and $_.self[1] -ge 190 -and $_.self[1] -le 450})
    if($crossing.Count -lt 10){throw 'Actual bridge traversal not proven'}
    foreach($row in $crossing) {
        $m=@($row.movers | Where-Object model -EQ 99)
        if($m.Count -ne 1 -or ($m[0].origin -join ',') -ne '0,-250,0' -or [math]::Abs($row.self[0]+512) -gt 16 -or !$row.on_ground -or $row.sent_command.Up -ne 0 -or $row.sent_command.Buttons -ne 0){throw 'Crossing lacks observed safe deployed support'}
    }
    if(!@($game | Where-Object {$_.self[1] -ge 500}).Count){throw 'Far bank not reached'}
    $last=$game[-1]
    $dx=$last.self[0]-$last.teammate[0];$dy=$last.self[1]-$last.teammate[1]
    $distance=[math]::Sqrt($dx*$dx+$dy*$dy)
    if($last.goal -ne 'cover_teammate' -or $distance -lt 96 -or $distance -gt 160){throw 'Spaced accompaniment after bridge not restored'}
    return [pscustomobject]@{crossing_frames=$crossing.Count;final_distance=$distance;health=$last.health;final_position=$last.self}
}
