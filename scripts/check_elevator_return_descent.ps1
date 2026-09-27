# Audit the native ledge return between the two base3 lift rides.
# This deliberately does not label a free fall as a ride on a descending mover.
function Assert-ElevatorReturnDescent($Rows) {
    $rows=@($Rows)
    $airIndex=-1
    for($i=1;$i -lt $rows.Count;$i++){
        if(!$rows[$i].on_ground -and $rows[$i-1].on_ground -and [math]::Abs($rows[$i-1].self[2]+231.875) -le 0.25){$airIndex=$i;break}
    }
    if($airIndex -lt 1){throw 'No departure from upper floor'}
    $start=$rows[$airIndex-1];$previous=$start;$air=0;$landingIndex=-1;$maxError=0.0
    for($i=$airIndex;$i -lt $rows.Count;$i++){
        $r=$rows[$i]
        if($r.frame -ne $previous.frame+1 -or $r.spawncount -ne $start.spawncount -or $r.health -ne 100){throw 'Invalid descent continuity or damage'}
        if($r.sent_command.Up -gt 0){throw 'Unexpected jump during ledge descent'}
        if($r.self[2] -gt $previous.self[2]+0.125){throw 'Upward motion in descent'}
        if($r.on_ground){$landingIndex=$i;break}
        if(!$previous.on_ground){
            # Native dry-air gravity: 800 units/s^2, one 100 ms server step.
            $velocity=$previous.self_velocity[2]-80
            $predicted=$previous.self[2]+$velocity*0.1
            $err=[math]::Abs($r.self[2]-$predicted)
            $maxError=[math]::Max($maxError,$err)
            if($err -gt 0.25 -or [math]::Abs($r.self_velocity[2]-$velocity) -gt 1){throw 'Descent does not match native gravity'}
        }
        $air++;$previous=$r
    }
    if($landingIndex -lt 0 -or $air -lt 5){throw 'Missing sustained free fall and landing'}
    $landing=$rows[$landingIndex]
    if([math]::Abs($landing.self[2]+423.875) -gt 0.25 -or [math]::Abs($start.self[2]-$landing.self[2]-192) -gt 0.25){throw 'Wrong lower-floor landing'}
    # The final step is clipped by collision with the lower floor.
    $predicted=$previous.self[2]+($previous.self_velocity[2]-80)*0.1
    if($predicted -gt $landing.self[2]+0.25 -or $landing.self[0] -lt 810 -or $landing.self[0] -gt 850 -or $landing.self[1] -lt -55 -or $landing.self[1] -gt -10){throw 'Landing does not match ledge geometry'}
    $settled=@($rows|Where-Object {$_.frame -ge $landing.frame -and $_.frame -le $landing.frame+3})
    if($settled.Count -ne 4 -or @($settled|Where-Object {!$_.on_ground -or $_.health -ne 100 -or [math]::Abs($_.self[2]+423.875) -gt 0.25}).Count){throw 'Landing support or health unstable'}
    $regroup=@($rows|Where-Object {$_.frame -gt $landing.frame -and $_.frame -le $landing.frame+20 -and $_.goal -eq 'cover_teammate' -and $_.on_ground -and [math]::Abs($_.self[2]+423.875) -le 0.25})
    if(!$regroup.Count){throw 'Did not regroup after descent'}
    [pscustomobject]@{mode='ledge_free_fall';departure_frame=$rows[$airIndex].frame;landing_frame=$landing.frame;air_frames=$air;drop_height=192;gravity_position_error=$maxError;landing_support_frames=4;health_loss=0}
}
