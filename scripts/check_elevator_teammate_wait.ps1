. "$PSScriptRoot/check_base2_elevator_exit.ps1"
function Assert-ElevatorTeammateWait($Rows,$Scenario,$ActorRows){
 $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
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
  if([math]::Abs($r.self[2]-24.125) -gt 0.25 -or $r.self[0] -lt -40 -or $r.self[0] -gt 0){throw 'Bot drifted off blocked exit'}
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
