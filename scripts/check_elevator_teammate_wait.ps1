. "$PSScriptRoot/check_base2_elevator_exit.ps1"
. "$PSScriptRoot/check_elevator_moving_teammate.ps1"
. "$PSScriptRoot/check_elevator_ascent.ps1"
. "$PSScriptRoot/check_elevator_late_controller.ps1"
function Assert-ElevatorTeammateWait($Rows,$Scenario,$ActorRows){
 if($Scenario.name -eq 'base2-elevator-late-controller'){return Assert-ElevatorLateController $Rows $Scenario $ActorRows}
 if($Scenario.name -like 'base2-elevator-ascent-*'){Assert-ElevatorAscentBlock $Rows $Scenario $ActorRows}
 if($Scenario.name -like 'base2-elevator-exit-teammate-cross*'){return Assert-ElevatorMovingTeammate $Rows $Scenario $ActorRows}
 $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
 $bypass=@($scene|Where-Object elevator -eq exit_teammate_bypass)
 if($Scenario.name -like '*side-walls*'){
  if($bypass.Count){throw 'Bypass attempted through side walls'}
  foreach($r in @($scene|Where-Object elevator -eq exit_teammate_wait)){
   foreach($y in @(-4,60)){
    $wall=@($r.movers|Where-Object {$_.model -eq 1 -and $_.origin[0] -eq -576 -and $_.origin[1] -eq $y -and $_.origin[2] -eq -24})
    if($wall.Count -ne 1){throw 'Missing observed side wall'}
   }
  }
 }
 if($bypass.Count){
  $exit=Assert-Base2ElevatorExit $Rows $Scenario
  if($bypass.Count -lt 5 -or $exit.completion_frame-$bypass[0].frame -gt 30){throw 'Incomplete bounded bypass'}
  $window=@($scene|Where-Object {$_.frame -ge $bypass[0].frame -and $_.frame -le $exit.completion_frame})
  $maxSide=0.0
  foreach($r in $window){
   if(!$r.on_ground -or !$r.teammate -or [math]::Abs($r.self[2]-24.125) -gt 0.25){throw 'Bypass lost supported landing'}
   $side=[math]::Abs($r.self[1]-1408);$maxSide=[math]::Max($maxSide,$side)
   if($side -gt 44 -or ([math]::Abs($r.self[0]-$r.teammate[0]) -lt 32 -and [math]::Abs($r.self[1]-$r.teammate[1]) -lt 32)){throw 'Bypass overlaps blocker or platform structure'}
   if([math]::Abs($r.teammate[0]-10) -gt 0.25 -or [math]::Abs($r.teammate[1]-1408) -gt 0.25){throw 'Blocker moved before bypass completed'}
  }
  if($maxSide -lt 36){throw 'No meaningful lateral bypass'}
  return [pscustomobject]@{accepted=$true;bypass_frames=$bypass.Count;lateral_offset=$maxSide;exit=$exit;mode='stationary_teammate_bypass'}
 }
 $wait=@($scene|Where-Object elevator -eq exit_teammate_wait)
 if($wait.Count -lt 25){throw 'Insufficient blocked-exit waiting'}
 $last=$null
 foreach($r in $scene){
  if($r.health -ne 100 -or ($last -and ($r.frame -ne $last.frame+1 -or $r.spawncount -ne $last.spawncount))){throw 'Damage or discontinuous wait episode'}
  $last=$r
 }
 foreach($r in $wait){
  $m=@($r.movers|Where-Object model -eq 50)
  if(!$r.on_ground -or $m.Count -ne 1 -or [math]::Abs($m[0].origin[2]) -gt 0.125 -or !$r.teammate){throw 'Unsafe wait state'}
  if($r.sent_command.Forward -ne 0 -or $r.sent_command.Side -ne 0 -or $r.sent_command.Up -ne 0){throw 'Bot pushing during wait'}
  if([math]::Abs($r.self[2]-24.125) -gt 0.25 -or $r.self[0] -lt -52 -or $r.self[0] -gt 0){throw 'Bot drifted off blocked exit'}
 }
 if($Scenario.name -like '*release'){
  $actor=@($ActorRows|Where-Object {$_.map -eq 'base2' -and $_.scenario.step_id -eq 'clear-lift-exit'})
  if($actor.Count -lt 3 -or $actor[-1].self[0]-$actor[0].self[0] -lt 40){throw 'Actor did not walk away'}
  for($i=1;$i -lt $actor.Count;$i++){if($actor[$i].frame -ne $actor[$i-1].frame+1 -or [math]::Abs($actor[$i].self[0]-$actor[$i-1].self[0]) -gt 32){throw 'Actor release was not continuous walking'}}
  $exit=Assert-Base2ElevatorExit $Rows $Scenario
  if($exit.completion_frame -le $wait[-1].frame -or $exit.completion_frame-$wait[-1].frame -gt 20){throw 'Did not resume promptly after release'}
  return [pscustomobject]@{accepted=$true;waiting_frames=$wait.Count;exit=$exit;mode='wait_then_resume'}
 }
 if(@($scene|Where-Object elevator -eq completed).Count -or $scene[-1].elevator -ne 'exit_teammate_wait'){throw 'False completion through blocker'}
 [pscustomobject]@{accepted=$true;waiting_frames=$wait.Count;mode='sustained_wait_without_pushing'}
}

