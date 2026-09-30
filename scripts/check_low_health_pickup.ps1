function Assert-LowHealthPickup([object[]]$Rows) {
    $game = @($Rows | Where-Object { $_.map -eq 'base2' -and $_.frame -ge 40 })
    $confirmed = @($game | Where-Object { $_.pickup.class -eq 'weapon_supershotgun' -and $_.pickup.state -eq 'confirmed' -and $_.pickup.before -eq 0 -and $_.pickup.after -eq 1 -and $_.health -eq 38 })
    if (!$confirmed.Count) { throw 'No confirmed Super Shotgun pickup at health 38' }
    if (@($game | Where-Object { $_.health -ne 38 -or $_.enemies.Count -gt 0 }).Count) { throw 'Unexpected damage, healing, death or enemy' }
    $before = @($game | Where-Object { $_.frame -le $confirmed[0].pickup.started_frame -and $_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.inventory | Where-Object name -EQ 'Super Shotgun').Count -eq 0 })
    if (!$before.Count) { throw 'Weapon missing before release not proven' }
    $last = $game[-1]
    $weapon = @($last.inventory | Where-Object name -EQ 'Super Shotgun')
    $shells = @($last.inventory | Where-Object name -EQ 'Shells')
    if ($weapon.Count -ne 1 -or $weapon[0].count -ne 1 -or $shells.Count -ne 1 -or $shells[0].count -lt 10 -or !$last.teammate -or $last.goal -ne 'cover_teammate') { throw 'Inventory gain or resumed follow missing' }
    $dx=$last.self[0]-$last.teammate[0]; $dy=$last.self[1]-$last.teammate[1]
    $distance=[math]::Sqrt($dx*$dx+$dy*$dy)
    if ($distance -gt 80) { throw 'Bot did not return to player' }
    return [pscustomobject]@{pickup_frame=$confirmed[0].frame;health=38;weapon_count=1;shells=$shells[0].count;final_distance=$distance}
}
