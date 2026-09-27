function Assert-ClosingDoorStop($Rows, $Fixture) {
    if($Fixture.model -ne 27 -or $Fixture.proximity_trigger -ne $false -or !$Fixture.native_door_physics){throw 'Wrong door fixture'}
    $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge 90 -and $_.frame -le 120})
    if($scene.Count -ne 31){throw 'Incomplete closing door trace'}
    $first=$null; $last=$null; $closing=0; $rest=0; $maxX=[double]::NegativeInfinity; $error=0.0
    foreach($r in $scene) {
        $door=@($r.movers|Where-Object model -eq 27)
        if($door.Count -ne 1 -or $r.health -ne 100 -or !$r.on_ground){throw 'Door observation, health or ground support lost'}
        $z=[double]$door[0].origin[2]
        if($last) {
            if($r.frame -ne $last.row.frame+1 -or $r.spawncount -ne $last.row.spawncount){throw 'Discontinuous door stop'}
            if($z -gt $last.z+0.125){throw 'Door reopened during approach'}
            if($z -lt $last.z-0.125){$closing++}
        }
        if(!$first -and $r.arbitration.move_limit_reason -eq 'dynamic_door_blocked') {
            if($r.self_velocity[0] -lt 290 -or !$last -or $z -ge $last.z){throw 'No full speed entry against closing door'}
            $first=$r
        }
        if($first) {
            if($r.arbitration.move_limit_reason -ne 'dynamic_door_blocked' -or $r.sent_command.Forward -ne 0 -or $r.sent_command.Side -ne 0 -or $r.sent_command.Up -ne 0){throw 'Movement continued after door block'}
            if($r.arbitration.move_source -eq 'ground_coast_brake' -or $r.arbitration.brake_input){throw 'Static brake applied to door'}
            if($last -and $last.row.frame -ge $first.frame) {
                if(!$last.row.ground_prediction){throw 'Missing coast prediction'}
                $pred=$last.row.ground_prediction
                for($axis=0;$axis -lt 3;$axis++) {
                    $e=[math]::Abs($r.self[$axis]-$last.row.self[$axis]-$pred.displacement[$axis])
                    $error=[math]::Max($error,$e)
                    if($e -gt 0.25 -or [math]::Abs($r.self_velocity[$axis]-$pred.velocity[$axis]) -gt 1){throw 'Observed stopping differs from free coast; possible contact'}
                }
            }
            $maxX=[math]::Max($maxX,$r.self[0])
            if([math]::Abs($r.self_velocity[0]) -lt 1 -and $z -le 0.125){$rest++}
        }
        $last=[pscustomobject]@{row=$r;z=$z}
    }
    $margin=$Fixture.standing_hull_boundary_x-$maxX
    if(!$first -or $closing -lt 5 -or $rest -lt 10 -or $margin -lt 8){throw 'Insufficient closing, stable stop or clearance'}
    [pscustomobject]@{accepted=$true;incoming_speed=$first.self_velocity[0];coast_distance=$maxX-$first.self[0];hull_margin=$margin;closed_rest_frames=$rest;prediction_error=$error;health_loss=0;scope='prepared_closing_door_neutral_coast_no_contact'}
}
