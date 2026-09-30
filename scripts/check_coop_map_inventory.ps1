function Assert-CoopMapInventory([object[]]$Rows, [string]$ServerLog, [string]$ObserverLog) {
    if (@($Rows.connection | Sort-Object -Unique).Count -ne 1) { throw 'Map transition replaced connection identity' }
    $first=@($Rows | Where-Object map -EQ 'base2')
    $next=@($Rows | Where-Object map -EQ 'base1')
    if (!$first.Count -or !$next.Count -or $first[-1].spawncount -eq $next[0].spawncount) { throw 'Missing map/generation boundary' }
    $acquired=@($first | Where-Object { $_.pickup.class -eq 'weapon_supershotgun' -and $_.pickup.state -eq 'confirmed' -and $_.pickup.before -eq 0 -and $_.pickup.after -eq 1 })
    if (!$acquired.Count) { throw 'Native weapon acquisition not proven' }
    $baseline=$first[-1]
    $shells=@($baseline.inventory | Where-Object name -EQ 'Shells')
    if ($shells.Count -ne 1 -or $shells[0].count -lt 10) { throw 'No baseline ammunition' }
    $death=@($next | Where-Object health -LE 0 | Select-Object -First 1)
    if (!$death.Count) { throw 'No death after transition' }
    $respawn=@($next | Where-Object { $_.frame -gt $death[0].frame -and $_.health -gt 0 } | Select-Object -First 1)
    if (!$respawn.Count) { throw 'No respawn' }
    $before=@($next | Where-Object { $_.frame -lt $death[0].frame -and $_.inventory_known -and $_.inventory_age_frames -le 20 })
    $after=@($next | Where-Object { $_.frame -ge $respawn[0].frame+10 -and $_.health -gt 0 -and $_.inventory_known -and $_.inventory_age_frames -le 20 })
    if ($before.Count -lt 3 -or $after.Count -lt 3) { throw 'Insufficient fresh inventory observations' }
    foreach($r in @($before)+@($after)) {
        $weapon=@($r.inventory | Where-Object name -EQ 'Super Shotgun')
        $ammo=@($r.inventory | Where-Object name -EQ 'Shells')
        if ($weapon.Count -ne 1 -or $weapon[0].count -ne 1 -or $ammo.Count -ne 1 -or $ammo[0].count -ne $shells[0].count) { throw 'Coop weapon/ammo lost' }
    }
    if ([regex]::Matches($ServerLog,'(?m)^GoCoopMate connected\s*$').Count -ne 1 -or [regex]::Matches($ServerLog,'(?m)^GoCoopMate entered the game\s*$').Count -ne 2) { throw 'Native server connection not preserved' }
    if ($ObserverLog -notmatch 'map signon resumed on existing UDP channel' -or $ObserverLog -match 'full reconnect started') { throw 'Unexpected transport reconnect' }
    return [pscustomobject]@{weapon='Super Shotgun';weapon_count=1;shells=$shells[0].count;fresh_before_death=$before.Count;fresh_after_respawn=$after.Count;death_frame=$death[0].frame;respawn_frame=$respawn[0].frame;connections=1;maps=@('base2','base1')}
}
