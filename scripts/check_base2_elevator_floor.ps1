. "$PSScriptRoot/check_base2_elevator_exit.ps1"
function Assert-Base2ElevatorFloor($Rows,$Scenario){
    $exit=Assert-Base2ElevatorExit $Rows $Scenario
    $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
    $hold=@($scene|Where-Object {$_.frame -le $Scenario.bot_release_frame})
    if($hold.Count -ne $Scenario.bot_release_frame-$Scenario.start_frame+1){throw 'Missing initial floor hold'}
    foreach($r in $hold){
        if(!$r.on_ground -or $r.self[0] -lt 0){throw 'Initial hull is not outside platform on floor'}
        for($i=0;$i -lt 3;$i++){if([math]::Abs($r.self[$i]-$Scenario.bot_origin[$i]) -gt 0.25){throw 'Initial floor position drift'}}
    }
    $ride=@($scene|Where-Object elevator -eq ride|Select-Object -First 1)
    if(!$ride.Count){throw 'No boarding from floor'}
    $window=@($scene|Where-Object {$_.frame -ge $ride[0].frame -and $_.frame -le $exit.completion_frame})
    $prev=$null;$rise=0.0;$rising=0;$support=0
    foreach($r in $window){
        if(!$r.on_ground){throw 'Lost support during lift ascent'}
        $m=@($r.movers|Where-Object model -eq 50)
        if($m.Count -ne 1){throw 'Missing platform50'}
        if($r.self[2] -lt 23.875){
            if($r.self[0] -lt -136 -or $r.self[0] -gt 0 -or $r.self[1] -lt 1352 -or $r.self[1] -gt 1464){throw 'Left platform before upper floor'}
            if($r.elevator -eq 'ride' -and $r.self[0] -gt -32){throw 'Ride hull not fully supported'}
            if([math]::Abs($r.self[2]-$m[0].origin[2]-24.125) -gt 0.25){throw 'Wrong deck support offset'}
            $support++
            if($prev){
                if($prev.id -ne $m[0].id){throw 'Platform identity changed'}
                $dz=$m[0].origin[2]-$prev.z
                if($dz -lt 0){throw 'Unexpected descending platform'}
                if($dz -gt 0.125){
                    if([math]::Abs($r.self[2]-$prev.selfZ-$dz) -gt 0.25){throw 'Uncorrelated ascent'}
                    $rise+=$dz;$rising++
                }
            }
        }
        $prev=@{id=$m[0].id;z=$m[0].origin[2];selfZ=$r.self[2]}
    }
    if($rise -lt 180 -or $rising -lt 10 -or $support -lt 15){throw 'Incomplete carried ascent'}
    [pscustomobject]@{accepted=$true;initial_floor_frames=$hold.Count;carried_height=$rise;rising_frames=$rising;supported_frames=$support;exit=$exit;scope='static_lower_floor_to_upper_landing'}
}
