function Assert-Jail3Descent([object[]]$Rows) {
    $placed=@($Rows | Where-Object {$_.map -eq 'jail3' -and [math]::Abs($_.self[0]+753.5) -lt 0.5 -and [math]::Abs($_.self[1]+648.75) -lt 0.5 -and [math]::Abs($_.self[2]-840.125) -lt 0.5})
    if(!$placed.Count){throw 'Captured staircase origin not reproduced'}
    $first=$placed[-1]
    if(!$first.teammate -or [math]::Abs($first.teammate[0]+336.125) -gt 0.5 -or [math]::Abs($first.teammate[1]+680.125) -gt 0.5 -or [math]::Abs($first.teammate[2]-776.125) -gt 0.5){throw 'Captured player position not reproduced'}
    $game=@($Rows | Where-Object {$_.map -eq 'jail3' -and $_.frame -ge $first.frame})
    if($game.Count -lt 100){throw 'Insufficient staircase observations'}
    if(@($game | Where-Object {$_.health -ne 7 -or !$_.teammate}).Count){throw 'Damage, healing, death or lost player'}
    $descent=@($game | Where-Object {$_.arbitration.move_source -eq 'walk_off' -and $_.self[0] -gt -753 -and $_.self[2] -lt 840})
    if(!$descent.Count){throw 'Verified staircase descent not observed'}
    if(!@($game | Where-Object {$_.frame -gt $first.frame -and $_.arbitration.skill -eq 'walk_off' -and $_.arbitration.limit_reason -eq 'drop_landed' -and $_.on_ground -and $_.self[2] -lt 840}).Count){throw 'Native descent landing not confirmed'}
    if(@($game | Where-Object {$_.arbitration.skill -eq 'walk_off' -and $_.arbitration.limit_reason -in @('drop_aborted','drop_missed')}).Count){throw 'Descent aborted or missed'}
    if(@($descent | Where-Object {$_.sent_command.Up -ne 0 -or $_.sent_command.Buttons -ne 0}).Count){throw 'Descent jumped or attacked'}
    if(!@($game | Where-Object {$_.self[0] -gt -620 -and $_.on_ground -and $_.self[2] -le 777}).Count){throw 'Lower landing not reached'}
    $last=$game[-1]
    $dx=$last.self[0]-$last.teammate[0];$dy=$last.self[1]-$last.teammate[1]
    $distance=[math]::Sqrt($dx*$dx+$dy*$dy)
    if($last.goal -ne 'cover_teammate' -or $distance -lt 96 -or $distance -gt 160){throw 'Spaced accompaniment after descent not restored'}
    return [pscustomobject]@{descent_frames=$descent.Count;final_distance=$distance;health=$last.health;final_position=$last.self}
}
