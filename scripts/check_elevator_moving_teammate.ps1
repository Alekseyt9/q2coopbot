function Assert-ElevatorMovingTeammate($Rows,$Scenario,$ActorRows){
 $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
 $bypass=@($scene|Where-Object elevator -eq exit_teammate_bypass)
 $wait=@($scene|Where-Object elevator -eq exit_teammate_wait)
 if($bypass.Count -lt 3 -or $wait.Count -lt 10 -or $wait[0].frame -le $bypass[0].frame){throw 'Missing bypass then blocked-path wait'}
 $actor=@($ActorRows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $bypass[0].frame -and $_.frame -le $wait[-1].frame})
 if($actor.Count -lt 10 -or 'cross-lift-exit' -notin $actor.scenario.step_id -or $actor[-1].self[1]-$actor[0].self[1] -lt 30){throw 'No actor crossing during bypass'}
 $last=$null
 foreach($r in $actor){
  if($last -and ($r.frame -ne $last.frame+1 -or [math]::Abs($r.self[0]-$last.self[0]) -gt 32 -or [math]::Abs($r.self[1]-$last.self[1]) -gt 32)){throw 'Actor crossing is not continuous walking'}
  $last=$r
 }
 $last=$null
 foreach($r in $scene){
  if($r.health -ne 100 -or ($last -and ($r.frame -ne $last.frame+1 -or $r.spawncount -ne $last.spawncount))){throw 'Damage or discontinuous crossing episode'}
  $last=$r
 }
 foreach($r in @($scene|Where-Object {$_.frame -ge $bypass[0].frame -and $_.frame -le $wait[-1].frame})){
  $m=@($r.movers|Where-Object model -eq 50)
  if(!$r.on_ground -or !$r.teammate -or [math]::Abs($r.self[2]-24.125) -gt 0.25 -or $m.Count -ne 1 -or [math]::Abs($m[0].origin[2]) -gt 0.125){throw 'Unsupported crossing state'}
  if([math]::Abs($r.self[1]-1408) -gt 44 -or ([math]::Abs($r.self[0]-$r.teammate[0]) -lt 32 -and [math]::Abs($r.self[1]-$r.teammate[1]) -lt 32)){throw 'Crossing overlaps blocker or structure'}
 }
 foreach($r in $wait){
  if($r.sent_command.Forward -ne 0 -or $r.sent_command.Side -ne 0 -or $r.sent_command.Up -ne 0){throw 'Bot pushes during blocked bypass'}
  if([math]::Abs($r.self[0]-$wait[0].self[0]) -gt 0.25 -or [math]::Abs($r.self[1]-$wait[0].self[1]) -gt 0.25){throw 'Bot drifts during blocked bypass'}
 }
 $exit=$null
 if($Scenario.name -like '*hold'){
  if('completed' -in $scene.elevator -or $scene[-1].elevator -ne 'exit_teammate_wait'){throw 'False completion of blocked bypass'}
 }else{
  $release=@($ActorRows|Where-Object {$_.scenario.step_id -eq 'clear-lift-exit'})
  if($release.Count -lt 3 -or $release[-1].self[0]-$release[0].self[0] -lt 40){throw 'Actor did not release exit'}
  $exit=Assert-Base2ElevatorExit $Rows $Scenario
  if($exit.completion_frame -le $wait[-1].frame -or $exit.completion_frame-$wait[-1].frame -gt 10){throw 'No prompt recovery after crossing'}
 }
 [pscustomobject]@{accepted=$true;mode=$Scenario.name;bypass_frames=$bypass.Count;waiting_frames=$wait.Count;exit=$exit}
}
