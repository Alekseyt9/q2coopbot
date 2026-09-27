. "$PSScriptRoot/check_elevator_boarding.ps1"
function Assert-ElevatorTrip($Rows,$Scenario) {
    $scene=@($Rows|Where-Object {$_.map -eq 'base3' -and $_.frame -ge $Scenario.bot_release_frame})
    $board=@($scene|Where-Object elevator -eq 'board'|Select-Object -First 1)
    $done=@($scene|Where-Object elevator -eq 'completed'|Select-Object -First 1)
    if(!$board.Count -or !$done.Count -or $done[0].frame -le $board[0].frame){throw 'Missing ordered boarding/completion'}
    $null=Assert-ElevatorBoarding $Rows $Scenario
    $ride=@($scene|Where-Object elevator -eq 'ride'|Select-Object -First 1)
    if(!$ride.Count){throw 'Missing ride phase'}
    $window=@($scene|Where-Object {$_.frame -ge $ride[0].frame -and $_.frame -le $done[0].frame+3})
    if($window.Count -ne $done[0].frame+4-$ride[0].frame){throw 'Incomplete trip window'}
    $previous=$null;$moverID=0;$rise=0.0;$rising=0;$supported=0;$stages=@('board');$rideStart=$null;$rideDrift=0.0
    foreach($r in $window){
        if($r.health -ne 100 -or !$r.on_ground){throw 'Damage or lost ground support during trip'}
        if($previous -and ($r.frame -ne $previous.frame+1 -or $r.spawncount -ne $previous.spawncount)){throw 'Discontinuous trip'}
        $m=@($r.movers|Where-Object model -eq 37)
        if($m.Count -ne 1){throw 'Missing platform observation'}
        if($moverID -and $moverID -ne $m[0].id){throw 'Platform identity changed'}
        $moverID=$m[0].id;$stages+= $r.elevator
        if($r.frame -lt $done[0].frame){
            if($r.self[0] -lt 776 -or $r.self[0] -gt 888 -or $r.self[1] -lt 288 -or $r.self[1] -gt 392){throw 'Body left deck before upper landing'}
            if([math]::Abs($r.self[2]-$m[0].origin[2]+231.875) -gt 0.25){throw 'Deck support height mismatch'}
            $supported++
            if($previous){
                $dz=$m[0].origin[2]-$previous.moverZ
                if($dz -lt -0.125){throw 'Platform descended during ascent'}
                if($dz -gt 0.125){
                    if([math]::Abs($r.self[2]-$previous.selfZ-$dz) -gt 0.25){throw 'Uncorrelated platform motion'}
                    $rise+=$dz;$rising++
                }
            }
        } else {
            if([math]::Abs($r.self[2]+231.875) -gt 0.25){throw 'Upper landing height mismatch'}
        }
        if($r.elevator -eq 'ride'){
            if(!$rideStart){$rideStart=$r.self}
            $d=[math]::Sqrt([math]::Pow($r.self[0]-$rideStart[0],2)+[math]::Pow($r.self[1]-$rideStart[1],2))
            $rideDrift=[math]::Max($rideDrift,$d)
            if($d -gt 24){throw 'Excessive drift while holding platform'}
        }
        $previous=[pscustomobject]@{frame=$r.frame;spawncount=$r.spawncount;moverZ=$m[0].origin[2];selfZ=$r.self[2]}
    }
    foreach($stage in @('board','ride','exit','completed')){if($stage -notin $stages){throw "Missing trip stage: $stage"}}
    if($rise -lt 180 -or $rising -lt 10 -or $supported -lt 15){throw 'Insufficient full ascent evidence'}
    if($window[-1].self[1] -gt 256){throw 'Body did not fully clear platform toward upper floor'}
    [pscustomobject]@{accepted=$true;completion_frame=$done[0].frame;supported_frames=$supported;rising_frames=$rising;carried_height=$rise;ride_drift=$rideDrift;exit_ground_frames=4;health_loss=0;scope='full_prepared_ascent_and_upper_exit'}
}

