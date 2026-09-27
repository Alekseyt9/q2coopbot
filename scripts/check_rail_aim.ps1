function Assert-RailAim($Rows,[string]$Mode) {
    $ErrorActionPreference='Stop'
    $rail=@($Rows | Where-Object {$_.weapon -like '*/v_rail/*' -and $_.health -gt 0 -and $_.map -eq 'base1'})
    if (!$rail.Count) {throw 'Railgun not confirmed'}
    $first=$rail[0]
    $target=@($first.enemies | Where-Object {$_.id -eq $first.arbitration.aim_entity -and $_.class -eq 'monster_tank'})
    if ($target.Count -ne 1 -or $first.ammo -ne 10) {throw 'Rail fixture target/ammo missing'}
    if ($Mode -eq 'rail_precision') {
        if ($first.arbitration.limit_reason -ne 'rail_aim_settling' -or ($first.sent_command.Buttons -band 1)) {throw 'Rail fired before acquisition'}
        $fire=@($rail | Where-Object {$_.frame -gt $first.frame -and $_.frame -le $first.frame+5 -and $_.arbitration.aim_entity -eq $target[0].id -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)} | Select-Object -First 1)
        if (!$fire.Count -or $fire[0].arbitration.aim_error_degrees -gt 0.5) {throw 'Accurate rail fire missing'}
        $previous=@($rail | Where-Object frame -eq ($fire[0].frame-1))
        if ($previous.Count -ne 1 -or $previous[0].arbitration.aim_entity -ne $target[0].id -or $previous[0].arbitration.aim_error_degrees -gt 0.5) {throw 'Two precise frames missing'}
        foreach ($r in @($previous[0],$fire[0])) {
            if (@($r.delta_angles).Count -ne 3 -or @($r.arbitration.aim_point).Count -ne 3) {throw 'Angle evidence missing'}
            $dx=$r.arbitration.aim_point[0]-$r.self[0];$dy=$r.arbitration.aim_point[1]-$r.self[1]
            $eye=$r.self[2]+ $(if($r.ducked){-2}else{22})
            $yaw=[math]::Atan2($dy,$dx)*180/[math]::PI
            $pitch=-[math]::Atan2($r.arbitration.aim_point[2]-$eye,[math]::Sqrt($dx*$dx+$dy*$dy))*180/[math]::PI
            foreach ($pair in @(@($yaw,$r.sent_command.Yaw,$r.delta_angles[1]),@($pitch,$r.sent_command.Pitch,$r.delta_angles[0]))) {
                $error=$pair[0]-($pair[1]+$pair[2])*360/65536
                $error=($error%360+540)%360-180
                if ([math]::Abs($error) -gt 0.5) {throw 'Sent rail angle is inaccurate'}
            }
        }
        $spent=@($rail | Where-Object {$_.frame -gt $fire[0].frame -and $_.frame -le $fire[0].frame+20 -and $_.ammo -lt 10})
        if (!$spent.Count) {throw 'No observed slug consumption'}
        return [pscustomobject]@{acquire_frame=$first.frame;fire_frame=$fire[0].frame;ammo_after=$spent[0].ammo;scope='acquisition_and_ammo_not_hit_rate'}
    }
    $window=@($rail | Where-Object {$_.frame -lt $first.frame+10})
    if ($window.Count -ne 10) {throw 'Behind-target guard window missing'}
    foreach ($r in $window) {
        $e=@($r.enemies | Where-Object id -eq $target[0].id)
        if ($e.Count -ne 1 -or @($r.teammate).Count -ne 3) {throw 'Behind-target setup missing'}
        $dx=$e[0].origin[0]-$r.self[0];$dy=$e[0].origin[1]-$r.self[1]
        $length2=$dx*$dx+$dy*$dy
        if ($length2 -lt 1) {throw 'Invalid target distance'}
        $projection=(($r.teammate[0]-$r.self[0])*$dx+($r.teammate[1]-$r.self[1])*$dy)/$length2
        $ex=$r.teammate[0]-$r.self[0]-$projection*$dx;$ey=$r.teammate[1]-$r.self[1]-$projection*$dy
        if ($projection -le 1 -or $ex*$ex+$ey*$ey -gt 24*24 -or [math]::Abs($r.teammate[2]-$r.self[2]) -gt 2) {throw 'Teammate is not behind target on the shot line'}
        if (($r.sent_command.Buttons -band 1) -or $r.ammo -ne 10 -or $r.arbitration.limit_reason -ne 'friendly_line_of_fire') {throw 'Rail behind-target guard failed'}
    }
    [pscustomobject]@{guarded_frames=10;ammo=10;scope='teammate_behind_target'}
}
