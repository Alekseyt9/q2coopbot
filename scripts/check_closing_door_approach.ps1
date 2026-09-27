function Assert-ClosingDoorApproach($Rows) {
    $scene=@($Rows | Where-Object { $_.map -eq 'base2' -and $_.frame -ge 80 -and $_.frame -le 120 })
    $previous=$null; $closing=0; $opening=0; $release=$null; $fast=0; $crossed=$false; $near=0
    foreach($r in $scene) {
        $doors=@($r.movers | Where-Object model -eq 27)
        if($doors.Count -ne 1){throw 'Door27 observation missing'}
        $z=[double]$doors[0].origin[2]
        if($r.health -ne 100 -or !$r.on_ground -or [math]::Abs($r.self[1]+800) -gt 2){throw 'Door approach damaged or displaced the bot'}
        if($previous -and ($r.frame -ne $previous.frame+1 -or $r.spawncount -ne $previous.spawncount)){throw 'Discontinuous door approach trace'}
        if(!$release -and $r.sent_command.Forward -gt 0) {
            if(!$previous -or $z -ge $previous.z-0.125 -or $closing -lt 3 -or [math]::Abs($r.self[0]+64) -gt 0.25){throw 'Bot was not released while the door was closing'}
            $release=$r.frame
        }
        if($previous -and $z -lt $previous.z-0.125){$closing++}
        if($release -and $previous -and $z -gt $previous.z+0.125){$opening++}
        if($r.ground_dynamic_model -eq 27) {
            $near++
            if($r.arbitration.move_source -eq 'ground_coast_brake' -or $r.arbitration.brake_input){throw 'Static brake applied near moving door'}
            if($r.self_velocity[0] -ge 250){$fast++}
        }
        if($release -and $r.self[0] -gt 96 -and $r.frame -le $release+15){$crossed=$true}
        $previous=[pscustomobject]@{frame=$r.frame;spawncount=$r.spawncount;z=$z}
    }
    if($scene.Count -ne 41 -or !$release -or $opening -lt 3 -or $fast -lt 2 -or !$crossed){throw 'Closing door did not reopen and permit prompt fast passage'}
    [pscustomobject]@{accepted=$true;release_frame=$release;closing_frames=$closing;reopening_frames=$opening;fast_near_frames=$fast;near_frames=$near;health_loss=0;crossed_grounded=$true;scope='closing_door_native_trigger_reopening_not_dynamic_braking'}
}
