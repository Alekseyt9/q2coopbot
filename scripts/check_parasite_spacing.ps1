function Measure-ParasiteSpacing($Rows,$Damage){
    $p=@($Rows|Where-Object {$_.arbitration.combat_spacing.enemy -eq 'monster_parasite'})
    if($p.Count -lt 20 -or @($p|Where-Object {$_.arbitration.combat_spacing.minimum -ne 320}).Count){throw 'Parasite profile not exercised'}
    foreach($r in $p){
        $e=@($r.enemies|Where-Object {$_.id -eq $r.arbitration.combat_spacing.target -and $_.class -eq 'monster_parasite'})
        if($e.Count -ne 1){throw 'Observed Parasite target missing'}
        $d=0.0;foreach($i in 0..2){$d+=($r.self[$i]-$e[0].origin[$i])*($r.self[$i]-$e[0].origin[$i])}
        if([math]::Abs([math]::Sqrt($d)-$r.arbitration.combat_spacing.distance) -gt 0.01){throw 'Reported distance disagrees with observed positions'}
    }
    $bot=@($Rows.self_entity|Sort-Object -Unique);if($bot.Count -ne 1 -or $bot[0] -lt 1){throw 'Ambiguous bot identity'}
    # The tongue starts 24 units ahead of the origin; 256 is not center reach.
    $escaped=@($p|Where-Object {$_.arbitration.combat_spacing.distance -gt 280})
    $retreat=@($p|Where-Object {$_.arbitration.move_source -eq 'combat_retreat'})
    $reposition=@($p|Where-Object {$_.arbitration.move_source -eq 'combat_firing_position'})
    $positionSteps=0
    $positionMin=1e9
    for($i=0;$i -lt $Rows.Count-1;$i++){
        $r=$Rows[$i];$next=$Rows[$i+1]
        if($r.arbitration.move_source -ne 'combat_firing_position'){continue}
        if($r.arbitration.combat_spacing.distance -lt 288 -or ($r.sent_command.Buttons -band 1)){throw 'Unsafe firing-position command'}
        if($next.frame -ne $r.frame+1 -or $next.spawncount -ne $r.spawncount){continue}
        $e=@($next.enemies|Where-Object {$_.id -eq $r.arbitration.combat_spacing.target})
        if($e.Count -ne 1){throw 'Missing threat after reposition'}
        $d=[math]::Sqrt([math]::Pow($next.self[0]-$e[0].origin[0],2)+[math]::Pow($next.self[1]-$e[0].origin[1],2))
        $positionMin=[math]::Min($positionMin,$d)
        if($d -le 280){throw 'Reposition entered tongue reach'}
        $movement=[math]::Sqrt([math]::Pow($next.self[0]-$r.self[0],2)+[math]::Pow($next.self[1]-$r.self[1],2))
        if($movement -gt 1){$positionSteps++}
    }
    $friendly=@($p|Where-Object {$_.arbitration.limit_reason -eq 'friendly_line_of_fire'})
    foreach($r in $friendly){if($r.sent_command.Buttons -band 1){throw 'Fired through teammate'}}
    $incoming=@($Damage|Where-Object {$_.target -eq $bot[0] -and $_.live_health_damage -gt 0})
    $outgoing=@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target_class -eq 'monster_parasite'})
    if(@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target_class -eq 'player' -and $_.take -gt 0}).Count){throw 'Server recorded friendly damage'}
    $kills=@($outgoing|Where-Object killed)
    $damageTaken=[int](($incoming|Measure-Object live_health_damage -Sum).Sum)
    [pscustomobject]@{escape_accepted=($escaped.Count -ge 10 -and $retreat.Count -ge 10 -and $damageTaken -eq 0 -and ($Rows.health|Measure-Object -Minimum).Minimum -gt 0);combat_completed=($kills.Count -gt 0);initial_distance=$p[0].arbitration.combat_spacing.distance;max_distance=($p.arbitration.combat_spacing.distance|Measure-Object -Maximum).Maximum;escaped_frames=$escaped.Count;retreat_commands=$retreat.Count;bot_damage=$damageTaken;target_damage=[int](($outgoing|Measure-Object live_health_damage -Sum).Sum);kills=$kills.Count;friendly_block_frames=$friendly.Count;reposition_commands=$reposition.Count;reposition_steps=$positionSteps;reposition_min_distance=$(if($positionSteps){$positionMin}else{$null});scope='prepared_single_parasite_native_ai'}
}

