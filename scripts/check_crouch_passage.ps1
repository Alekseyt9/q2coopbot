function Assert-NaturalCrouchPassage($Rows) {
    $start=@($Rows | Where-Object {$_.frame -ge 40 -and $_.on_ground -and !$_.ducked -and [math]::Abs($_.self[0]+510) -lt 2 -and [math]::Abs($_.self[1]+35) -lt 1 -and [math]::Abs($_.self[2]+23.875) -lt 1} | Select-Object -First 1)
    if (!$start.Count) {throw 'Natural crouch: standing entry setup missing'}
    $request=@($Rows | Where-Object {$_.frame -gt $start[0].frame -and $_.arbitration.skill -eq 'crouch_passage' -and $_.sent_command.Up -lt 0 -and $_.self[0] -le -486} | Select-Object -First 1)
    if (!$request.Count) {throw 'Natural crouch: pre-entry duck request missing'}
    $exit=@($Rows | Where-Object {$_.frame -gt $request[0].frame -and !$_.ducked -and $_.on_ground -and $_.self[0] -gt -422} | Select-Object -First 1)
    if (!$exit.Count) {throw 'Natural crouch: standing exit missing'}
    $window=@($Rows | Where-Object {$_.frame -ge $start[0].frame -and $_.frame -le $exit[0].frame})
    $inside=@($window | Where-Object {$_.self[0] -ge -478 -and $_.self[0] -le -430})
    if ($inside.Count -lt 5) {throw 'Natural crouch: did not traverse low section'}
    foreach ($r in $inside) {
        if (!$r.ducked -or $r.sent_command.Up -ge 0 -or $r.arbitration.skill -ne 'crouch_passage') {throw 'Natural crouch: posture not held under ceiling'}
    }
    for ($i=0;$i -lt $window.Count;$i++) {
        $r=$window[$i]
        if ($r.map -ne 'base1' -or $r.health -le 0 -or !$r.on_ground -or $r.sent_command.Up -gt 0 -or [math]::Abs($r.self[1]+35) -gt 2 -or [math]::Abs($r.self[2]+23.875) -gt 1) {throw 'Natural crouch: left verified corridor or jumped/died'}
        if ($i -gt 0 -and ($r.frame -ne $window[$i-1].frame+1 -or $r.self[0]-$window[$i-1].self[0] -gt 40 -or $r.self[0] -lt $window[$i-1].self[0]-1)) {throw 'Natural crouch: discontinuous traversal'}
    }
    [pscustomobject]@{entry_frame=$start[0].frame;request_frame=$request[0].frame;exit_frame=$exit[0].frame;duck_samples=$inside.Count}
}

function Assert-TooTightCrouchRefusal($Rows) {
    # Require a real safe sidestep and restored following, not a frozen refusal.
    $start=@($Rows | Where-Object frame -eq 40)
    if ($start.Count -ne 1 -or !$start[0].on_ground -or $start[0].ducked -or [math]::Abs($start[0].self[0]+510) -gt 1 -or [math]::Abs($start[0].self[1]+42) -gt 1 -or [math]::Abs($start[0].self[2]+23.875) -gt 1) {throw 'Tight passage: wrong standing setup'}
    $anchor=@($Rows | Where-Object {$_.frame -gt 40 -and $_.frame -le 44 -and $_.on_ground -and !$_.ducked -and [math]::Abs($_.self[0]+510) -lt 2 -and [math]::Abs($_.self[1]+26) -lt 2} | Select-Object -First 1)
    if (!$anchor.Count) {throw 'Tight passage: safe reconnect step missing or delayed'}
    $cover=@($Rows | Where-Object {$_.frame -gt $anchor[0].frame -and $_.frame -le 60 -and $_.goal -eq 'cover_teammate' -and $_.self[0] -gt -350 -and $_.on_ground} | Select-Object -First 1)
    if (!$cover.Count) {throw 'Tight passage: timely follow recovery missing'}
    $window=@($Rows | Where-Object {$_.frame -ge 40 -and $_.frame -le $cover[0].frame})
    if ($window.Count -ne $cover[0].frame-39) {throw 'Tight passage: incomplete recovery trace'}
    for ($i=0;$i -lt $window.Count;$i++) {
        $r=$window[$i]
        if ($r.frame -ne 40+$i -or $r.map -ne 'base1' -or $r.health -le 0 -or $r.ducked -or $r.sent_command.Up -ne 0 -or $r.arbitration.skill -eq 'crouch_passage' -or [math]::Abs($r.self[2]+23.875) -gt 18) {throw 'Tight passage: unsafe recovery'}
        if (@($r.teammate).Count -ne 3 -or [math]::Abs($r.teammate[0]+270) -gt 2 -or [math]::Abs($r.teammate[1]+42) -gt 1) {throw 'Tight passage: target changed'}
        if ($i -gt 0) {
            $dx=$r.self[0]-$window[$i-1].self[0];$dy=$r.self[1]-$window[$i-1].self[1]
            if ($dx*$dx+$dy*$dy -gt 41*41) {throw 'Tight passage: position discontinuity'}
        }
    }
    [pscustomobject]@{anchor_frame=$anchor[0].frame;cover_frame=$cover[0].frame;scope='safe_reconnect_and_follow_without_jump'}
}
