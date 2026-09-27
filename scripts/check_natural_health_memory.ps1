# Require a new choice while the remembered entity is absent, then real healing.
# Retaining an already visible health goal is not a memory-selection proof.
function Assert-NaturalHealthMemory($Rows, [int]$MinHealth) {
    if (@($Rows | Where-Object test_health_masked).Count) { throw 'Natural memory test used a perception mask' }
    for ($i=1; $i -lt $Rows.Count; $i++) {
        $choice=$Rows[$i]; $previous=$Rows[$i-1]
        if ($choice.goal -ne 'recover_health' -or $choice.health -le 0 -or @($choice.goal_point).Count -ne 3) { continue }
        $target=$choice.goal_point
        if ($previous.goal -eq 'recover_health' -and ($previous.goal_point -join ',') -eq ($target -join ',')) { continue }
        foreach ($memory in $choice.resource_memory) {
            if ($memory.item.class -ne 'item_health' -or $memory.state -ne 'unknown' -or !$memory.attempted) { continue }
            $item=$memory.item
            if ([math]::Abs($item.origin[0]-$target[0]) -ge 1 -or [math]::Abs($item.origin[1]-$target[1]) -ge 1 -or [math]::Abs($item.origin[2]+9.125-$target[2]) -ge 1) { continue }
            if (@($choice.pickups | Where-Object id -eq $item.id).Count) { continue }
            if (!@($Rows | Where-Object { $_.frame -lt $choice.frame -and @($_.pickups | Where-Object id -eq $item.id).Count }).Count) { continue }
            $seenAgain=$false
            for ($j=$i+1; $j -lt $Rows.Count; $j++) {
                $row=$Rows[$j]; $prev=$Rows[$j-1]
                if ($row.health -le 0 -or $row.map -ne $choice.map) { break }
                if (@($row.pickups | Where-Object id -eq $item.id).Count) { $seenAgain=$true }
                if (!$seenAgain -or $row.health -lt $MinHealth -or $row.health -le $prev.health -or $prev.goal -ne 'recover_health' -or ($prev.goal_point -join ',') -ne ($target -join ',')) { continue }
                if ([math]::Abs($row.self[0]-$target[0]) -lt 48 -and [math]::Abs($row.self[1]-$target[1]) -lt 48 -and [math]::Abs($row.self[2]-$target[2]) -lt 32) { return $row.frame }
            }
        }
    }
    throw 'Natural unseen health choice, reobservation and server healing not confirmed'
}
