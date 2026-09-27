function Measure-CombatRunIn($Rows,$Damage){
    $warm=@($Rows|Where-Object {$_.arbitration.move_source -eq 'test_combat_run_in'})
    if($warm.Count -ne 3){throw 'Expected exactly3 run-in commands'}
    $handoff=@($Rows|Where-Object {$_.frame -eq $warm[-1].frame+1 -and $_.spawncount -eq $warm[-1].spawncount})
    if($handoff.Count -ne 1 -or $handoff[0].arbitration.move_source -ne 'combat_retreat'){throw 'Combat control did not resume immediately'}
    $h=$handoff[0];$distance=0.0
    foreach($axis in 0..2){$distance+=[math]::Pow($h.self[$axis]-$warm[-1].self[$axis],2)}
    $speed=[math]::Sqrt($distance)*10
    if(!$h.PSObject.Properties['self_velocity'] -or @($h.self_velocity).Count -ne 3){throw 'Server velocity missing at handoff'}
    $serverSpeed=[math]::Sqrt([math]::Pow($h.self_velocity[0],2)+[math]::Pow($h.self_velocity[1],2))
    if($serverSpeed -lt 250){throw 'Run-in displacement did not retain incoming server velocity'}
    $enemy=@($h.enemies|Where-Object {$_.id -eq $h.arbitration.combat_spacing.target})
    if($enemy.Count -ne 1){throw 'Missing handoff threat'}
    $approach=0.0;$before=0.0;$after=0.0
    foreach($axis in 0..2){$before+=[math]::Pow($warm[-1].self[$axis]-$enemy[0].origin[$axis],2);$after+=[math]::Pow($h.self[$axis]-$enemy[0].origin[$axis],2)}
    $approach=[math]::Sqrt($before)-[math]::Sqrt($after)
    $afterRows=@($Rows|Where-Object {$_.frame -ge $h.frame -and $_.frame -le $h.frame+3})
    $minDistance=1e9
    foreach($r in $afterRows){$d=0.0;foreach($axis in 0..2){$d+=[math]::Pow($r.self[$axis]-$enemy[0].origin[$axis],2)};$minDistance=[math]::Min($minDistance,[math]::Sqrt($d))}
    $overshoot=[math]::Sqrt($after)-$minDistance
    $bot=@($Rows.self_entity|Sort-Object -Unique);if($bot.Count -ne 1){throw 'Ambiguous bot'}
    $taken=[int]((@($Damage|Where-Object target -eq $bot[0])|Measure-Object live_health_damage -Sum).Sum)
    if(@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target_class -eq 'player' -and $_.take -gt 0}).Count){throw 'Friendly damage'}
    $kills=@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target_class -eq 'monster_parasite' -and $_.killed})
    $escaped=@($Rows|Where-Object {$_.arbitration.combat_spacing.distance -gt 280})
    [pscustomobject]@{accepted=($speed -ge 250 -and $approach -ge 25 -and $overshoot -le 8 -and $escaped.Count -ge 10 -and $kills.Count -eq 1 -and $taken -eq 0);handoff_speed=$speed;server_handoff_speed=$serverSpeed;handoff_approach=$approach;braking_overshoot=$overshoot;escaped_frames=$escaped.Count;bot_damage=$taken;kills=$kills.Count;scope='prepared_three_tick_run_in_native_parasite'}
}
