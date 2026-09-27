function Measure-MovingTeammate($Rows,$Actor,$Damage){
    $bot=@($Rows.self_entity|Sort-Object -Unique);$friend=@($Actor.self_entity|Sort-Object -Unique)
    if($bot.Count -ne 1 -or $friend.Count -ne 1 -or $bot[0] -eq $friend[0]){throw 'Ambiguous client identities'}
    $actorFrames=@{};foreach($r in $Actor){$actorFrames["$($r.spawncount):$($r.frame)"]=$r}
    $steps=0;$blockedMoving=0;$distance=0.0
    $blocked=@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'friendly_line_of_fire'})
    $predicted=@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'friendly_projectile_crossing'})
    foreach($r in $predicted){if($r.sent_command.Buttons -band 1){throw 'Fired during predicted crossing'}}
    foreach($r in $blocked){if($r.sent_command.Buttons -band 1){throw 'Fired through teammate'}}
    for($i=1;$i -lt $Rows.Count;$i++){
        $r=$Rows[$i];$prev=$Rows[$i-1]
        if(!$r.teammate -or !$prev.teammate -or $r.frame -ne $prev.frame+1 -or $r.spawncount -ne $prev.spawncount){continue}
        $a=$actorFrames["$($r.spawncount):$($r.frame)"]
        if(!$a){continue}
        $d=0.0
        foreach($axis in 0..2){
            if([math]::Abs($a.self[$axis]-$r.teammate[$axis]) -gt 0.125){throw 'Client observations disagree on actor position'}
            $d+=[math]::Pow($r.teammate[$axis]-$prev.teammate[$axis],2)
        }
        $d=[math]::Sqrt($d)
        if($d -gt 1 -and $a.arbitration.move_source -eq 'test_walk'){
            $steps++;$distance+=$d
            if($r.arbitration.limit_reason -eq 'friendly_line_of_fire'){$blockedMoving++}
        }
    }
    foreach($a in $Actor|Where-Object {$_.arbitration.limit_reason -eq 'test_combat_barrier'}){
        if($a.sent_command.Forward -or $a.sent_command.Side -or $a.sent_command.Up){throw 'Actor moved before combat release'}
    }
    $friendly=@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target -eq $friend[0] -and $_.take -gt 0})
    if($friendly.Count){throw 'Server recorded damage to teammate'}
    $incoming=@($Damage|Where-Object {$_.target -in @($bot[0],$friend[0])})
    $taken=[int](($incoming|Measure-Object live_health_damage -Sum).Sum)
    $kills=@($Damage|Where-Object {$_.attacker -eq $bot[0] -and $_.target_class -eq 'monster_parasite' -and $_.killed})
    $resumed=@($Rows|Where-Object {$blocked.Count -gt 0 -and $_.frame -gt $blocked[-1].frame -and ($_.sent_command.Buttons -band 1)})
    [pscustomobject]@{accepted=($steps -ge 4 -and $distance -ge 100 -and $blockedMoving -ge 1 -and $resumed.Count -ge 3 -and $kills.Count -eq 1 -and $taken -eq 0);actor_observed_steps=$steps;actor_distance=$distance;moving_blocked_frames=$blockedMoving;blocked_frames=$blocked.Count;predicted_block_frames=$predicted.Count;resumed_fire_frames=$resumed.Count;kills=$kills.Count;players_damage=$taken;scope='prepared_moving_teammate_single_parasite'}
}
