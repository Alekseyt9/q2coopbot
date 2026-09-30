function Test-NetworkReturn($Rows,$Events,$Signal,$Memory,[int]$DurationMS=1500) {
    $trigger=@($Events|Where-Object trigger)
    if($trigger.Count -ne 1){throw 'Missing or repeated network trigger'}
    $t=$trigger[0]
    if($t.trigger.map -ne $Signal.map -or $t.trigger.generation -ne $Signal.generation -or $t.trigger.frame -ne $Signal.frame -or $t.trigger.role -ne 'observer'){throw 'Wrong network trigger identity'}
    $blackout=@($Events|Where-Object stage -eq 'blackout')
    # A frame-paced client stops sending movement when snapshots stop. Zero
    # upstream packets is expected; every packet that does arrive must be lost.
    $down=@($blackout|Where-Object direction -eq 'server_to_client')
    $up=@($blackout|Where-Object direction -eq 'client_to_server')
    if($down.Count -lt 20){throw 'No sustained snapshot packet loss'}
    foreach($stage in @('unarmed','after')){
        if(!@($Events|Where-Object {$_.stage -eq $stage -and $_.direction -eq 'client_to_server' -and $_.action -eq 'forward'}).Count){throw 'No upstream traffic outside loss interval'}
    }
    if(@($blackout|Where-Object action -ne 'drop').Count -or @($Events|Where-Object {$_.stage -ne 'blackout' -and $_.action -ne 'forward'}).Count){throw 'Loss does not match scheduled interval'}
    if(($blackout[-1].elapsed_ms-$blackout[0].elapsed_ms) -lt $DurationMS-200){throw 'Blackout too short'}
    if(@($Events|Where-Object decode_error).Count){throw 'Relay decode failure'}
    $lostFrames=@($down|ForEach-Object {$_.frames}|ForEach-Object {$_.frame})
    if(!$lostFrames.Count -or @($Rows|Where-Object {$_.frame -in $lostFrames}).Count){throw 'Lost snapshots leaked into bot observations'}
    $active=@($Rows|Where-Object {$_.frame -le $Signal.frame -and $_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0 -and !$_.teammate -and !$_.last_teammate})
    if($active.Count -lt 20 -or $active[-1].frame -ne $Signal.frame){throw 'Loss was not armed during sustained death return'}
    $firstAfter=@($Events|Where-Object {$_.stage -eq 'after' -and $_.direction -eq 'server_to_client' -and $_.frames.Count}|Select-Object -First 1)
    if(!$firstAfter.Count){throw 'No server snapshots after loss'}
    $afterFrame=$firstAfter[0].frames[0].frame
    $after=@($Rows|Where-Object {$_.frame -ge $afterFrame})
    $before=@($Rows|Where-Object {$_.frame -le $Signal.frame}|Select-Object -Last 1)
    if($after.Count -lt 10 -or ($after[0].frame-$before[0].frame) -lt [math]::Floor($DurationMS/50)-4){throw 'Missing observed snapshot gap or recovery window'}
    $death=$Memory.death
    if(!$death -or !$Memory.completed -or $Memory.player -or $Memory.map -ne $Signal.map -or $Memory.generation -ne $Signal.generation){throw 'Return completion was not preserved'}
    $gameplay=@($Rows|Where-Object {$_.frame -ge $active[0].frame})
    if(@($gameplay|Where-Object {$_.health -le 0 -or $_.teammate -or $_.last_teammate -or $_.map -ne $Signal.map -or $_.spawncount -ne $Signal.generation}).Count){throw 'Death/contact/map change contaminated recovery'}
    $previous=0
    foreach($row in $gameplay){
        if($row.frame -le $previous){throw 'Nonmonotonic recovery frames'}
        $previous=$row.frame
        if($row.goal -eq 'regroup_after_respawn'){
            foreach($axis in 0..2){if(!$row.goal_point -or [math]::Abs($row.goal_point[$axis]-$death[$axis]) -gt .125){throw 'Return target changed across loss'}}
        }
    }
    $tail=@($after|Select-Object -Last 10)
    foreach($row in $tail){
        $distance=[math]::Sqrt([math]::Pow($row.self[0]-$death[0],2)+[math]::Pow($row.self[1]-$death[1],2))
        if($row.goal -ne 'wait_for_teammate' -or !$row.on_ground -or $distance -gt 64 -or [math]::Abs($row.self[2]-$death[2]) -gt 40){throw 'Bot did not settle at remembered death point'}
    }
    if(@($after|Where-Object goal -eq 'regroup_after_respawn').Count -lt 3){throw 'No resumed return after loss'}
    @{trigger_frame=$Signal.frame;recovery_frame=$after[0].frame;snapshot_gap=$after[0].frame-$before[0].frame;blackout_packets=$blackout.Count;downstream_drops=$down.Count;upstream_drops=$up.Count;recovery_frames=$after.Count;arrival_frame=$tail[0].frame;completed=$true}
}
