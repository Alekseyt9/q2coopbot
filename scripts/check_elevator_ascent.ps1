function Assert-ElevatorAscentBlock($Rows,$Scenario,$ActorRows){
 $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
 $actor=@($ActorRows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
 $walking=@($actor|Where-Object {$_.scenario.step_id -eq 'block-during-ascent'})
 if($walking.Count -lt 3 -or $walking[0].self[0]-$walking[-1].self[0] -lt 50){throw 'Actor did not walk into exit'}
 $last=$null
 foreach($r in $actor){
  if($last -and ($r.frame -ne $last.frame+1 -or [math]::Abs($r.self[0]-$last.self[0]) -gt 32)){throw 'Discontinuous ascent blocker movement'}
  $last=$r
 }
 $rising=0;$blocked=0;$near=0;$lastZ=$null
 foreach($r in $scene){
  $m=@($r.movers|Where-Object model -eq 50)
  if($m.Count -ne 1){throw 'Missing ascent platform'}
  $z=$m[0].origin[2]
  if($null -ne $lastZ -and $z -gt $lastZ -and $z -lt -0.125){
   $rising++
   if(!$r.on_ground -or $r.elevator -notin @('ride','exit','landing_probe') -or $r.sent_command.Up -gt 0){throw 'Unsafe controller state during ascent'}
   if($r.teammate -and $r.teammate[0] -ge 0 -and $r.teammate[0] -le 30 -and [math]::Abs($r.teammate[1]-1408) -lt 1){
    $blocked++
    if([math]::Abs($r.self[0]-$r.teammate[0]) -lt 64 -and [math]::Abs($r.self[2]-$r.teammate[2]) -lt 48){$near++}
   }
  }
  $lastZ=$z
 }
 if($rising -lt 8 -or $blocked -lt 5 -or $near -lt 1){throw 'No occupied exit during observed ascent'}
 if(@($scene|Where-Object elevator -eq exit_teammate_bypass).Count){throw 'Unexpected bypass in ascent block fixture'}
}
