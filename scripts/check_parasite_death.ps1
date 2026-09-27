function Assert-ParasiteDeath($Rows,$Damage) {
    $kills=@($Damage | Where-Object {$_.target_class -eq 'monster_parasite' -and $_.killed})
    if($kills.Count -ne 1){throw 'Expected one real parasite death'}
    $kill=$kills[0]
    $seen=@($Rows | Where-Object {$_.spawncount -eq $kill.spawncount -and $_.frame -ge $kill.frame -and @($_.defeated | Where-Object {$_.id -eq $kill.target -and $_.class -eq 'monster_parasite' -and $_.frame -ge 32 -and $_.frame -le 38}).Count})
    if($seen.Count -lt 7 -or !@($seen | Where-Object {@($_.defeated | Where-Object {$_.id -eq $kill.target -and $_.frame -eq 38}).Count}).Count){throw 'Complete visible death animation not observed'}
    $after=@($Rows | Where-Object {$_.spawncount -eq $kill.spawncount -and $_.frame -ge $seen[0].frame})
    if($after.Count -lt 20){throw 'Insufficient post-death observation'}
    foreach($r in $after) {
        if(@($r.enemies | Where-Object id -eq $kill.target).Count -or $r.arbitration.aim_entity -eq $kill.target){throw 'Corpse remains a combat target'}
        if(!@($r.enemies | Where-Object { $null -ne $_ }).Count -and ($r.sent_command.Buttons -band 1)){throw 'Attack continues without enemies'}
    }
    [pscustomobject]@{accepted=$true;death_frame=$kill.frame;corpse_frames=$seen.Count;target=$kill.target}
}
