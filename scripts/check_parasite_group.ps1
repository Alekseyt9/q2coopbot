function Measure-ParasiteGroup($Rows,$Damage){
    $ids=@($Rows.self_entity|Sort-Object -Unique)
    if($ids.Count -ne 1 -or $ids[0] -lt 1){throw 'Ambiguous bot identity'}
    $bot=$ids[0]
    $group=@($Rows|Where-Object {@($_.enemies|Where-Object {$_.class -eq 'monster_parasite' -and $_.clear_shot -eq $true}).Count -eq 2})
    $targets=@($group.enemies|Where-Object class -eq monster_parasite|Select-Object -ExpandProperty id -Unique)
    if($group.Count -lt 10 -or $targets.Count -ne 2){throw 'Two visible native threats not exercised'}
    $away=0;$maxLeash=0.0
    for($i=0;$i -lt $Rows.Count-1;$i++){
        $r=$Rows[$i];$next=$Rows[$i+1]
        if($r.arbitration.limit_reason -eq 'friendly_line_of_fire' -and ($r.sent_command.Buttons -band 1)){throw 'Fired through teammate'}
        if(!$r.teammate){throw 'Missing teammate observation'}
        $leash=0.0;foreach($axis in 0..2){$leash+=[math]::Pow($r.self[$axis]-$r.teammate[$axis],2)}
        $maxLeash=[math]::Max($maxLeash,[math]::Sqrt($leash))
        if($r.arbitration.move_source -ne 'combat_retreat' -or $next.frame -ne $r.frame+1){continue}
        $enemies=@($r.enemies|Where-Object {$_.clear_shot -eq $true})
        if($enemies.Count -ne 2){continue}
        $safe=$true;$moved=$false
        foreach($e in $enemies){
            $before=0.0;$after=0.0
            foreach($axis in 0..2){$before+=[math]::Pow($r.self[$axis]-$e.origin[$axis],2);$after+=[math]::Pow($next.self[$axis]-$e.origin[$axis],2)}
            $delta=[math]::Sqrt($after)-[math]::Sqrt($before)
            if($delta -lt -2){$safe=$false}
            if($delta -gt 1){$moved=$true}
        }
        if($safe -and $moved){$away++}
    }
    $incoming=@($Damage|Where-Object target -eq $bot)
    $taken=[int](($incoming|Measure-Object live_health_damage -Sum).Sum)
    $outgoing=@($Damage|Where-Object attacker -eq $bot)
    if(@($outgoing|Where-Object {$_.target_class -eq 'player' -and $_.take -gt 0}).Count){throw 'Friendly damage on server'}
    $kills=@($outgoing|Where-Object {$_.target -in $targets -and $_.killed}|Select-Object -ExpandProperty target -Unique)
    $last=$Rows[-1];$endDistance=0.0
    foreach($axis in 0..2){$endDistance+=[math]::Pow($last.self[$axis]-$last.teammate[$axis],2)}
    $endDistance=[math]::Sqrt($endDistance)
    [pscustomobject]@{accepted=($taken -eq 0 -and $kills.Count -eq 2 -and $away -ge 3 -and $maxLeash -le 360 -and $endDistance -le 220 -and $last.health -gt 0);visible_group_frames=$group.Count;observed_group_away_steps=$away;bot_damage=$taken;kills=$kills.Count;max_teammate_distance=$maxLeash;final_teammate_distance=$endDistance;scope='prepared_two_parasites_native_ai_no_model'}
}
