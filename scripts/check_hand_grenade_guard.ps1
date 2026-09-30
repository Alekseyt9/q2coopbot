function Assert-HandGrenadeGuard([object[]]$Rows, [object]$Summary) {
    $hand = @($Rows | Where-Object weapon -Match '/v_handgr/')
    if (!$hand.Count) { throw 'Native hand grenade was never selected' }
    $after = @($Rows | Where-Object { $_.frame -ge $hand[0].frame })
    if (@($hand | Where-Object { ($_.sent_command.Buttons -band 1) -ne 0 }).Count) { throw 'Hand grenade attack held' }
    foreach ($row in $after) {
        $grenades = @($row.inventory | Where-Object name -EQ 'Grenades')
        if (!$row.inventory_known -or $row.inventory_age_frames -gt 20 -or $grenades.Count -ne 1 -or $grenades[0].count -ne 5) { throw 'Grenades not conserved with fresh inventory' }
        if ($row.health -le 0) { throw 'Bot died' }
    }
    $shots = @($after | Where-Object { $_.weapon -eq 'Blaster' -and ($_.sent_command.Buttons -band 1) -ne 0 -and $_.enemies.Count -gt 0 })
    if ($shots.Count -lt 3) { throw 'Blaster combat did not resume' }
    if ($Summary.timescale -ne 2 -or $Summary.decode_errors -ne 0 -or $Summary.frame_gaps -ne 0) { throw 'Invalid native trace' }
    return [pscustomobject]@{hand_frames=$hand.Count;hand_attack_frames=0;grenades_remaining=5;blaster_combat_frames=$shots.Count;final_health=$after[-1].health}
}
