function Assert-ClosingDoorPhase($Rows,$Fixture,[int]$Release,[string]$Expected,[double]$ExpectedSpeed=300,[double]$CoastMargin=8,$ExpectedX=$null) {
    $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Release -and $_.frame -le 120})
    if($scene.Count -ne 121-$Release){throw 'Incomplete phase trace'}
    if($null -ne $ExpectedX -and ([math]::Abs($scene[0].self[0]-$ExpectedX) -gt 0.25 -or [math]::Abs($scene[0].self_velocity[0]) -gt 1)){throw 'Wrong approach start'}
    $last=$null;$minMargin=[double]::PositiveInfinity;$brakes=0;$closed=0;$fast=0;$crossed=$false;$error=0.0
    foreach($r in $scene) {
        $doors=@($r.movers|Where-Object model -eq 27)
        if($doors.Count -ne 1 -or $r.health -ne 100 -or !$r.on_ground){throw 'Missing door, damage or loss of support'}
        $z=$doors[0].origin[2]
        if($r.arbitration.move_source -eq 'ground_coast_brake'){throw 'Static brake at door'}
        if($last) {
            if($r.frame -ne $last.row.frame+1 -or $r.spawncount -ne $last.row.spawncount -or $z -gt $last.z+0.125){throw 'Discontinuity or door reversal'}
            $p=$last.row.ground_prediction
            if(!$p){throw 'Missing command prediction'}
            for($a=0;$a -lt 3;$a++){
                $e=[math]::Abs($r.self[$a]-$last.row.self[$a]-$p.displacement[$a]);$error=[math]::Max($error,$e)
                if($e -gt 0.25 -or [math]::Abs($r.self_velocity[$a]-$p.velocity[$a]) -gt 1){throw 'Contact or command prediction mismatch'}
            }
        }
        if($r.self_velocity[0] -gt $ExpectedSpeed+2){throw 'Approach exceeded requested speed'}
        if($r.self_velocity[0] -ge $ExpectedSpeed-2){$fast++}
        if($r.self[0] -gt 96){$crossed=$true}
        $minMargin=[math]::Min($minMargin,$Fixture.standing_hull_boundary_x-$r.self[0])
        if($z -le 0.125 -and [math]::Abs($r.self_velocity[0]) -lt 1){$closed++}
        if($r.arbitration.move_source -in @('door_coast_brake','door_approach_brake')){
            $brakes++
            if($r.arbitration.move_source -ne $Expected){throw 'Unexpected brake type'}
            $input=$r.arbitration.brake_input
            if(!$input){throw 'Missing pre-brake command'}
            foreach($key in @('Pitch','Yaw','Roll','Buttons','Impulse','Up','Msec')){if($input.$key -ne $r.sent_command.$key){throw "Brake changed $key"}}
            if($Expected -eq 'door_coast_brake' -and ($input.Forward -ne 0 -or $input.Side -ne 0)){throw 'Coast brake overrides active move'}
            if($Expected -eq 'door_approach_brake' -and $input.Forward -eq 0 -and $input.Side -eq 0){throw 'Approach not active'}
        }
        $last=[pscustomobject]@{row=$r;z=$z}
    }
    if($fast -lt 2 -or $closed -lt 10){throw 'No fast entry or stable closed-door result'}
    if($Expected -eq 'pass') {if(!$crossed -or $brakes){throw 'Safe passage unnecessarily stopped'}}
    else {
        if($crossed -or $minMargin -lt $(if($Expected -eq 'coast'){$CoastMargin}else{1})){throw 'Insufficient stopping margin'}
        if($brakes -ne $(if($Expected -eq 'coast'){0}else{1})){throw 'Wrong brake count'}
    }
    [pscustomobject]@{accepted=$true;release_frame=$Release;expected=$Expected;expected_speed=$ExpectedSpeed;speed_frames=$fast;brakes=$brakes;minimum_hull_margin=$(if($Expected -ne 'pass'){$minMargin}else{$null});prediction_error=$error;health_loss=0;closed_rest_frames=$closed}
}
