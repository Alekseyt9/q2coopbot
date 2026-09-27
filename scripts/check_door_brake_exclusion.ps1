function Assert-DoorBrakeExclusion($Rows){
    $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and [math]::Abs($_.self[1]+800) -lt 64})
    $near=@($scene|Where-Object {$_.ground_dynamic_model -eq 27})
    if($near.Count -lt 3){throw 'Door27 exclusion not exercised'}
    if(@($near|Where-Object {$_.arbitration.move_source -eq 'ground_coast_brake' -or $_.arbitration.brake_input}).Count){throw 'Static brake applied near dynamic door'}
    $moving=0;$last=$null;$minZ=[double]::PositiveInfinity;$maxZ=[double]::NegativeInfinity
    foreach($r in $scene){
        $door=@($r.movers|Where-Object model -eq 27)
        if($door.Count -ne 1){$last=$null;continue}
        $z=[double]$door[0].origin[2];$minZ=[math]::Min($minZ,$z);$maxZ=[math]::Max($maxZ,$z)
        if($last -and $r.frame -eq $last.frame+1 -and $r.spawncount -eq $last.spawncount -and [math]::Abs($z-$last.z) -gt 0.125){$moving++}
        $last=[pscustomobject]@{frame=$r.frame;spawncount=$r.spawncount;z=$z}
    }
    $blocked=@($near|Where-Object {$_.arbitration.move_limit_reason -eq 'dynamic_door_blocked' -and $_.sent_command.Forward -eq 0 -and $_.sent_command.Side -eq 0})
    $crossed=@($scene|Where-Object {$_.self[0] -gt 96 -and $_.on_ground})
    if($moving -lt 3 -or $minZ -gt 8 -or $maxZ -lt 72 -or !$blocked.Count -or !$crossed.Count){throw 'Door did not move, block, open and permit grounded crossing'}
    [pscustomobject]@{accepted=$true;dynamic_model=27;near_frames=$near.Count;moving_frames=$moving;blocked_frames=$blocked.Count;min_z=$minZ;max_z=$maxZ;crossed_grounded=$true;static_brake_frames=0;scope='opening_door_static_brake_exclusion_not_high_speed_door_braking'}
}
