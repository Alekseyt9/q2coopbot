function Assert-WallCoastBrake($Rows,[switch]$Edge,[switch]$East,$ExpectedOrigin=$null,$ExpectedVelocity=$null){
    if($East -and !$Edge){throw 'East variant requires edge mode'}
    $warm=@($Rows|Where-Object {$_.arbitration.move_source -eq 'test_combat_run_in'})
    $commands=if($Edge){1}else{3}
    if($warm.Count -ne $commands){throw 'Coast brake run-in command count mismatch'}
    $brakes=@($Rows|Where-Object {$_.arbitration.move_source -eq 'ground_coast_brake'})
    if($brakes.Count -ne 1){throw "Require exactly one brake, got $($brakes.Count)"}
    $r=$brakes[0]
    $x=if($Edge){460}else{-34}
    if($East){$x=364}
    $y=if($Edge){140}else{-224}
    $z=if($Edge){-39.875}else{24.125}
    $hazard=if($Edge){'uneven_or_missing_support'}else{'static_hull_blocked'}
    $vx=if($East){300}else{-300}
    $vy=0
    if($null -ne $ExpectedOrigin){
        if(@($ExpectedOrigin).Count -ne 3 -or @($ExpectedVelocity).Count -ne 3){throw 'Incomplete expected approach'}
        foreach($value in @($ExpectedOrigin)+@($ExpectedVelocity)){
            if($null -eq $value -or $value -is [string] -or $value -is [bool] -or [double]::IsNaN([double]$value) -or [double]::IsInfinity([double]$value)){throw 'Invalid expected approach value'}
        }
        $x,$y,$z=$ExpectedOrigin;$vx,$vy,$vz=$ExpectedVelocity
        if([math]::Abs($vz) -gt 0.001 -or [math]::Abs([math]::Sqrt($vx*$vx+$vy*$vy)-300) -gt 0.01){throw 'Expected horizontal run-in speed must be300'}
    }
    if($r.frame -ne $warm[-1].frame+1 -or [math]::Abs($r.self[0]-$x) -gt 0.25 -or [math]::Abs($r.self[1]-$y) -gt 0.25 -or [math]::Abs($r.self[2]-$z) -gt 0.25 -or [math]::Abs($r.self_velocity[0]-$vx) -gt 0.25 -or [math]::Abs($r.self_velocity[1]-$vy) -gt 0.25){throw 'Coast approach not reproduced'}
    $inputCmd=$r.arbitration.brake_input
    if(!$inputCmd -or $inputCmd.Forward -ne 0 -or $inputCmd.Side -ne 0 -or $inputCmd.Up -ne 0){throw 'Brake input was not a neutral command'}
    foreach($key in @('Pitch','Yaw','Roll','Buttons','Impulse','Msec','Up')){
        if($inputCmd.$key -ne $r.sent_command.$key){throw "Brake changed $key"}
    }
    $yaw=([double]$inputCmd.Yaw+[double]$r.delta_angles[1])*2*[math]::PI/65536
    $incomingSpeed=[math]::Sqrt($vx*$vx+$vy*$vy)
    $alignment=[math]::Abs(($vx*[math]::Cos($yaw)+$vy*[math]::Sin($yaw))/$incomingSpeed)
    if($Edge -and $null -eq $ExpectedOrigin -and $alignment -gt 0.95){throw 'Edge fixture did not exercise movement across the view direction'}
    if($r.ground_prediction.neutral_path -ne $hazard -or $r.ground_prediction.command_then_stop_path -ne 'static_sampled_clear'){throw 'Brake did not replace risky coasting with a clear stop'}
    $next=@($Rows|Where-Object {$_.frame -eq $r.frame+1 -and $_.spawncount -eq $r.spawncount -and $_.map -eq $r.map})
    if($next.Count -ne 1){throw 'Missing server observation after brake'}
    $shift=[math]::Sqrt([math]::Pow($next[0].self[0]-$r.self[0],2)+[math]::Pow($next[0].self[1]-$r.self[1],2))
    $speed=[math]::Sqrt([math]::Pow($next[0].self_velocity[0],2)+[math]::Pow($next[0].self_velocity[1],2))
    if($shift -gt 0.25 -or $speed -gt 1 -or !$next[0].on_ground -or [math]::Abs($next[0].self[2]-$z) -gt 0.25){throw "Did not stop on supported floor: shift=$shift speed=$speed"}
    $tail=@($Rows|Where-Object {$_.frame -gt $r.frame -and $_.frame -le $r.frame+10})
    if($tail.Count -ne 10 -or @($tail|Where-Object {!$_.on_ground -or [math]::Abs($_.self[2]-$z) -gt 0.25 -or [math]::Abs($_.self[0]-$r.self[0]) -gt 0.25 -or [math]::Abs($_.self[1]-$r.self[1]) -gt 0.25 -or $_.ground_surface -ne 'dry_flat'}).Count){throw 'Stop did not persist for ten frames'}
    [pscustomobject]@{accepted=$true;incoming_speed=[math]::Sqrt([math]::Pow($r.self_velocity[0],2)+[math]::Pow($r.self_velocity[1],2));view_velocity_alignment=$alignment;aim_preserved=$true;stopping_shift=$shift;remaining_speed=$speed;support_margin=$(if($Edge){$null}else{$next[0].self[0]+48});scope=$(if($Edge){'prepared_neutral_coast_full_footprint_edge_100ms'}else{'prepared_neutral_coast_static_wall_100ms'})}
}
