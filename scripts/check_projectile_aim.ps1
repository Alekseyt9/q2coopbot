function Assert-ProjectileAim($Rows,[string]$Mode) {
    $ErrorActionPreference='Stop'
    $spawn=@($Rows | ForEach-Object {$_.enemies} | Where-Object {$_.class -eq 'monster_soldier' -and $_.origin[0] -ge 1060 -and $_.origin[0] -le 1120 -and $_.origin[1] -ge 330 -and $_.origin[1] -le 390 -and [math]::Abs($_.origin[2]+32) -lt 4} | Select-Object -First 1)
    if (!$spawn.Count) {throw 'Native moving soldier fixture not observed'}
    $id=$spawn[0].id
    $byFrame=@{};foreach($r in $Rows){$byFrame[[int]$r.frame]=$r}
    $count=0
    foreach ($r in $Rows) {
        $a=$r.arbitration
        $weaponOK=if ($Mode -eq 'projectile_hyper') {$r.weapon -like '*/v_hyperb/*'} else {$r.weapon -eq 'Blaster'}
        if (!$weaponOK -or $a.aim_entity -ne $id -or $a.lead_seconds -le 0 -or !($r.sent_command.Buttons -band 1)) {continue}
        $now=@($r.enemies | Where-Object id -eq $id)
        $last=@($byFrame[([int]$r.frame-1)].enemies | Where-Object id -eq $id)
        $older=@($byFrame[([int]$r.frame-2)].enemies | Where-Object id -eq $id)
        if ($now.Count -ne 1 -or $last.Count -ne 1 -or $older.Count -ne 1 -or !$now[0].clear_shot -or $r.map -ne 'base1' -or $r.health -le 0) {throw 'Lead without consecutive observations'}
        $e=$now[0];$t=[double]$a.lead_seconds
        $vx=10*($e.origin[0]-$last[0].origin[0]);$vy=10*($e.origin[1]-$last[0].origin[1])
        $ox=10*($last[0].origin[0]-$older[0].origin[0]);$oy=10*($last[0].origin[1]-$older[0].origin[1])
        $speed=[math]::Sqrt($vx*$vx+$vy*$vy)
        if ($speed -lt 10 -or $speed -gt 400 -or ($vx-$ox)*($vx-$ox)+($vy-$oy)*($vy-$oy) -gt 6400.01 -or $t -gt 0.75 -or $speed*$t -gt 128) {throw 'Unstable or unbounded lead'}
        $top=8*(($e.solid -shr 10) -band 63)-32
        $bottom=-8*(($e.solid -shr 5) -band 31)
        $height=[math]::Min([math]::Max(22,$bottom+8),$top-8)
        $expected=@(($e.origin[0]+$vx*$t),($e.origin[1]+$vy*$t),($e.origin[2]+$height))
        $distance2=0.0
        for($i=0;$i -lt 3;$i++) {
            if ([math]::Abs($a.aim_point[$i]-$expected[$i]) -gt 0.02) {throw 'Aim does not match measured motion'}
            $eye=$r.self[$i];if($i -eq 2){$eye+= $(if($r.ducked){-2}else{22})}
            $distance2+=($a.aim_point[$i]-$eye)*($a.aim_point[$i]-$eye)
        }
        if ([math]::Abs([math]::Sqrt($distance2)-1000*$t) -gt 0.02) {throw 'Intercept flight time mismatch'}
        $count++
    }
    if($count -lt 2){throw 'Fewer than two verified lead fire commands for the intended weapon/target'}
    [pscustomobject]@{target=$id;verified_lead_commands=$count;scope='command_geometry_not_hit_rate'}
}
