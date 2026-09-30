function Assert-PlayerPersonalSpace([object[]]$Rows) {
    $game = @($Rows | Where-Object { $_.map -eq 'base2' -and $_.frame -ge 40 })
    if ($game.Count -lt 100) { throw 'Insufficient native observations' }
    if (@($game | Where-Object {$_.health -ne 22 -or !$_.teammate -or [math]::Abs($_.self[2]+71.875) -gt 1}).Count) { throw 'Damage, lost player or unsupported retreat' }
    $initial = $game[0]
    if ([math]::Abs($initial.self[0]-424.375) -gt 1 -or [math]::Abs($initial.self[1]+2369.625) -gt 1 -or [math]::Abs($initial.teammate[1]+2401.75) -gt 1) { throw 'Original blocking position not reproduced' }
    $yielded = @($game | Where-Object { $_.arbitration.move_source -eq 'teammate_yield' -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0) })
    if (!$yielded.Count) { throw 'No applied yielding policy' }
    $hold = @($game | Where-Object { $_.frame -ge 100 -and $_.frame -le 140 })
    if ($hold.Count -lt 20) { throw 'Stable spacing interval missing' }
    foreach ($row in $hold + @($game[-1])) {
        $dx=$row.self[0]-$row.teammate[0]; $dy=$row.self[1]-$row.teammate[1]
        $d=[math]::Sqrt($dx*$dx+$dy*$dy)
        if ($d -lt 96 -or $d -gt 160) { throw 'Personal space not maintained' }
    }
    $last=$game[-1]
    if ($last.teammate[1] -le $initial.self[1]+40 -or $last.goal -ne 'cover_teammate') { throw 'Player did not pass the original blocker or accompaniment not restored' }
    return [pscustomobject]@{yield_frames=$yielded.Count;player_advance=$last.teammate[1]-$initial.teammate[1];final_distance=$d;health=$last.health}
}
