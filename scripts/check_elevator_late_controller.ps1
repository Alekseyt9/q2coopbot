function Assert-ElevatorLateController($Rows,$Scenario,$ActorRows){
 $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
 $previous=$null
 foreach($r in $scene){
  if($r.health -ne 100 -or ($previous -and ($r.frame -ne $previous.frame+1 -or $r.spawncount -ne $previous.spawncount))){throw 'Late controller damage or discontinuity'}
  if($r.elevator -eq 'wait_bottom' -and $r.self[2] -gt 20){throw 'Waiting for bottom while already upstairs'}
  $previous=$r
 }
 $ride=@($scene|Where-Object {$_.goal -eq 'cover_teammate' -and !$_.elevator -and $_.self[2] -lt 20})
 if($ride.Count -lt 5 -or ($ride.self|ForEach-Object {$_[2]}|Measure-Object -Maximum).Maximum -lt -20){throw 'Missing shared ride before controller activation'}
 $walk=@($ActorRows|Where-Object {$_.scenario.step_id -eq 'leave-platform'})
 if($walk.Count -lt 5 -or $walk[-1].self[0]-$walk[0].self[0] -lt 180){throw 'Actor did not leave by walking'}
 $done=@($scene|Where-Object elevator -eq completed|Select-Object -First 1)
 if($done.Count -ne 1 -or $done[0].frame-$walk[0].frame -gt 35){throw 'No prompt exit from shared ride'}
 $settled=@($scene|Where-Object {$_.frame -ge $done[0].frame+10 -and $_.frame -le $done[0].frame+40})
 if($settled.Count -ne 31){throw 'Missing settled exit window'}
 foreach($r in $settled){if(!$r.on_ground -or $r.self[0] -lt 0 -or [math]::Abs($r.self[2]-24.125) -gt .25){throw 'Did not clear platform and settle'}}
 [pscustomobject]@{accepted=$true;completion_frame=$done[0].frame;settled_frames=$settled.Count;health_loss=0}
}
