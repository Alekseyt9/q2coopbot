function Assert-DropRisk([object[]]$Rows,[int]$InitialHealth,[string]$Map='base1') {
    $source=if($Map -eq 'base1'){@(-937,1149,344.125)}else{@(831,-574,-231.875)}
    $player=if($Map -eq 'base1'){@(-850,1292,136.125)}else{@(980,-574,-487.875)}
    $estimate=if($Map -eq 'base1'){7}else{11}
    $placed=@($Rows|Where-Object {$_.map -eq $Map -and $_.arbitration.limit_reason -ne 'test_setup_hold' -and [math]::Abs($_.self[0]-$source[0]) -lt 0.5 -and [math]::Abs($_.self[1]-$source[1]) -lt 0.5 -and [math]::Abs($_.self[2]-$source[2]) -lt 0.5})
    if(!$placed.Count){throw 'Drop source not reproduced'}
    $first=$placed[0];$game=@($Rows|Where-Object {$_.map -eq $Map -and $_.frame -ge $first.frame})
    if($game.Count -lt 100 -or $first.health -ne $InitialHealth -or $first.gravity -ne 800 -or !$first.on_ground -or !$first.teammate){throw 'Invalid native drop setup'}
    if([math]::Abs($first.teammate[0]-$player[0]) -gt 0.5 -or [math]::Abs($first.teammate[1]-$player[1]) -gt 0.5 -or [math]::Abs($first.teammate[2]-$player[2]) -gt 0.5){throw 'Lower player position not reproduced'}
    if(@($game|Where-Object {$_.health -le 0 -or $_.health -gt $InitialHealth -or $_.test_health_masked -or $_.gravity -ne 800}).Count){throw 'Invalid health/gravity or masked observation'}
    $plans=@($game|Where-Object {$_.jump_plan.drop -and $_.jump_plan.expected_damage -gt 0})
    if($InitialHealth -eq 7) {
        if($plans.Count -or @($game|Where-Object {$_.health -ne 7 -or $_.self[2] -lt $source[2]-64}).Count){throw 'Critical-health bot accepted damaging descent'}
        return [pscustomobject]@{initial_health=7;final_health=7;damaging_drop_rejected=$true}
    }
    if(!$plans.Count -or @($plans|Where-Object {$_.jump_plan.expected_damage -ne $estimate}).Count){throw 'Conservative damage budget not observed'}
    if(@($game|Where-Object {$_.arbitration.skill -eq 'walk_off' -and ($_.sent_command.Up -ne 0 -or $_.sent_command.Buttons -ne 0)}).Count){throw 'Walk-off jumped or attacked'}
    if(!@($game|Where-Object {$_.arbitration.limit_reason -eq 'drop_landed' -and $_.on_ground -and [math]::Abs($_.self[2]-$player[2]) -lt 4}).Count){throw 'Lower native landing not confirmed'}
    $last=$game[-1];$damage=$InitialHealth-$last.health
    $minimumDamage=if($Map -eq 'base3'){1}else{0}
    if($damage -lt $minimumDamage -or $damage -gt $estimate -or $last.health -lt 25 -or $last.goal -ne 'cover_teammate' -or !$last.teammate){throw 'Damage budget or accompaniment violated'}
    $dx=$last.self[0]-$last.teammate[0];$dy=$last.self[1]-$last.teammate[1];$distance=[math]::Sqrt($dx*$dx+$dy*$dy)
    if($distance -lt 96 -or $distance -gt 160 -or [math]::Abs($last.self[2]-$last.teammate[2]) -gt 18){throw 'Player not regained on lower floor'}
    return [pscustomobject]@{map=$Map;initial_health=$InitialHealth;final_health=$last.health;predicted_damage=$estimate;actual_damage=$damage;final_distance=$distance}
}
