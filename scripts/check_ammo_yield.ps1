function Assert-AmmoYield($Rows, $ActorRows, $Expected) {
    foreach ($row in $Rows) {
        if ($row.resource_yield.class -ne $Expected.class -or $row.resource_yield.reason -ne 'teammate_closer' -or !$row.inventory_known -or $row.inventory_age_frames -gt 20 -or $row.health -le 0) {continue}
        $entity=$row.resource_yield.entity
        $item=@($row.pickups|Where-Object { $_.id -eq $entity -and $_.class -eq $Expected.class })
        if ($item.Count -ne 1) {continue}
        $matches=$true
        for ($axis=0;$axis -lt 3;$axis++) {if ([math]::Abs($item[0].origin[$axis]-$Expected.origin[$axis]) -gt 2) {$matches=$false}}
        if (!$matches) {continue}
        if (!@($row.inventory|Where-Object { $_.name -eq $Expected.weapon -and $_.count -gt 0 }).Count) {continue}
        $before=@($ActorRows|Where-Object { $_.frame -le $row.frame -and $_.inventory_known -and $_.inventory_age_frames -le 20 -and $_.health -gt 0 }|Select-Object -Last 1)
        if (!$before.Count -or $row.frame-$before[0].frame -gt 20) {continue}
        $baseline=0
        foreach ($entry in $before[0].inventory) {if ($entry.name -eq $Expected.name) {$baseline=$entry.count}}
        if (@($Rows|Where-Object { $_.frame -ge $row.frame -and $_.pickup.entity -eq $entity -and $_.pickup.state -in @('approach','confirmed') }).Count) {continue}
        foreach ($actor in $ActorRows) {
            if ($actor.frame -le $row.frame -or !$actor.inventory_known -or $actor.inventory_age_frames -gt 20 -or $actor.health -le 0) {continue}
            if (!@($actor.inventory|Where-Object { $_.name -eq $Expected.name -and $_.count -ge $baseline+$Expected.gain }).Count) {continue}
            if ([math]::Abs($actor.self[0]-$item[0].origin[0]) -ge 48 -or [math]::Abs($actor.self[1]-$item[0].origin[1]) -ge 48 -or [math]::Abs($actor.self[2]-$item[0].origin[2]-9.125) -ge 32) {continue}
            if (@($ActorRows|Where-Object { $_.frame -ge $before[0].frame -and $_.frame -le $actor.frame -and ($_.health -le 0 -or $_.map -ne $row.map) }).Count) {continue}
            return $actor.frame
        }
    }
    throw 'Ammo yield with useful bot inventory and actor inventory gain not confirmed'
}
