. "$PSScriptRoot/check_elevator_return_descent.ps1"
. "$PSScriptRoot/check_elevator_trip.ps1"
function Assert-ElevatorCycle($Rows,$Scenario) {
    $scene=@($Rows|Where-Object {$_.map -eq 'base3' -and $_.frame -ge $Scenario.bot_release_frame})
    $done=@($scene|Where-Object elevator -eq completed)
    if($done.Count -ne 2){throw 'Expected exactly two completed ascents'}
    $last=$null
    foreach($r in $scene){
        if($r.health -ne 100){throw 'Damage in isolated lift cycle'}
        if(@($r.enemies | Where-Object { $_ }).Count){throw 'Combat entities present in isolated fixture'}
        if($last -and ($r.frame -ne $last.frame+1 -or $r.spawncount -ne $last.spawncount)){throw 'Cycle observation discontinuity'}
        $last=$r
    }
    $first=Assert-ElevatorTrip $Rows $Scenario
    $secondBoard=@($scene|Where-Object {$_.frame -gt $done[0].frame+3 -and $_.elevator -eq 'board'}|Select-Object -First 1)
    if(!$secondBoard.Count){throw 'Missing second boarding'}
    $again=$Scenario|ConvertTo-Json -Depth 12|ConvertFrom-Json
    $again.bot_release_frame=$secondBoard[0].frame-10
    if($again.bot_release_frame -le $done[0].frame+3){throw 'No separated return interval'}
    $second=Assert-ElevatorTrip $Rows $again
    if($second.completion_frame -ne $done[1].frame){throw 'Wrong second trip'}
    $between=@($scene|Where-Object {$_.frame -gt $done[0].frame+3 -and $_.frame -lt $secondBoard[0].frame})
    $returnDescent=Assert-ElevatorReturnDescent $between
    $floor=@($between|Where-Object {$_.on_ground -and [math]::Abs($_.self[2]+423.875) -le 0.25 -and $_.self[1] -lt 256})
    if($floor.Count -lt 30){throw 'Insufficient lower-floor waiting'}
    $top=$false;$bottom=$false;$descent=0.0;$lastMover=$null
    foreach($r in $between){
        $m=@($r.movers|Where-Object model -eq 37)
        if($m.Count -ne 1){$lastMover=$null;continue}
        $z=$m[0].origin[2]
        if([math]::Abs($z) -le 0.125){$top=$true}
        if($lastMover -and $lastMover.frame+1 -eq $r.frame -and $lastMover.id -eq $m[0].id -and $z -lt $lastMover.z){$descent+=$lastMover.z-$z}
        if($top -and [math]::Abs($z+190) -le 0.125){$bottom=$true}
        $lastMover=[pscustomobject]@{frame=$r.frame;id=$m[0].id;z=$z}
    }
    if(!$bottom -or $descent -lt 180){throw 'Platform return not observed'}
    $hold=@($scene|Where-Object {$_.frame -ge $done[1].frame+40})
    if($hold.Count -lt 30 -or @($hold|Where-Object {!$_.on_ground -or [math]::Abs($_.self[2]+231.875) -gt 0.25 -or $_.goal -ne 'cover_teammate'}).Count){throw 'Upper-floor regroup/hold failed'}
    [pscustomobject]@{accepted=$true;return_descent=$returnDescent;first=$first;second=$second;lower_wait_frames=$floor.Count;return_height=$descent;upper_hold_frames=$hold.Count;scope='two_ascents_native_return_without_combat'}
}

