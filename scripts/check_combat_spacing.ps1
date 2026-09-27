function Assert-CombatSpacing($Rows){
    $retreat=@($Rows|Where-Object {$_.arbitration.move_source -eq 'combat_retreat'})
    $hold=@($Rows|Where-Object {$_.arbitration.move_limit_reason -eq 'combat_spacing_hold'})
    if($retreat.Count -lt 3 -or $hold.Count -lt 2){throw 'Retreat and settled combat spacing were not exercised'}
    $byFrame=@{};foreach($r in $Rows){$byFrame[[int]$r.frame]=$r}
    $moved=0;$fired=0
    foreach($r in $retreat){
        $p=$r.arbitration.combat_spacing
        if(!$p.need_space -or $p.distance -ge $p.minimum -or !$r.on_ground -or $r.sent_command.Up -ne 0){throw 'Invalid retreat state'}
        if(($r.sent_command.Buttons -band 1) -and $r.arbitration.aim_source -eq 'enemy'){$fired++}
        $next=$byFrame[([int]$r.frame+1)]
        $e=@($r.enemies|Where-Object id -eq $p.target)
        if(!$next -or $next.map -ne $r.map -or $e.Count -ne 1){throw 'Retreat observation missing'}
        # Compare both player positions to the same observed target position:
        # the enemy running away must not count as the bot's retreat.
        $before=0.0;$after=0.0
        foreach($i in 0..1){$before+=($r.self[$i]-$e[0].origin[$i])*($r.self[$i]-$e[0].origin[$i]);$after+=($next.self[$i]-$e[0].origin[$i])*($next.self[$i]-$e[0].origin[$i])}
        if([math]::Sqrt($after)-[math]::Sqrt($before) -gt 2){$moved++}
    }
    foreach($r in $hold){if($r.sent_command.Forward -or $r.sent_command.Side -or $r.sent_command.Up){throw 'Follow movement fought the spacing hold'}}
    if($moved -lt 3 -or $fired -lt 3){throw 'Retreat was not applied while aiming/firing'}
    [pscustomobject]@{retreat_commands=$retreat.Count;observed_away_steps=$moved;retreat_fire_commands=$fired;hold_frames=$hold.Count;scope='prepared_single_soldier_group_guard_unit_tested'}
}
