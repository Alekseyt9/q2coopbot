function Assert-Base2ElevatorExit($Rows,$Scenario) {
    $scene=@($Rows|Where-Object {$_.map -eq 'base2' -and $_.frame -ge $Scenario.start_frame})
    if(!$scene.Count){throw 'No base2 observations'}
    $last=$null
    foreach($r in $scene){
        if($r.health -ne 100){throw "Lift exit damage: frame=$($r.frame) health=$($r.health)"}
        if($last -and ($r.frame -ne $last.frame+1 -or $r.spawncount -ne $last.spawncount)){throw 'Discontinuous lift exit observations'}
        $last=$r
    }
    foreach($stage in @('ride','exit','completed')){if($stage -notin $scene.elevator){throw "Missing lift stage: $stage"}}
    $done=@($scene|Where-Object elevator -eq completed|Select-Object -First 1)[0]
    $settled=@($scene|Where-Object {$_.frame -ge $done.frame+10 -and $_.frame -le $done.frame+30})
    if($settled.Count -ne 21){throw 'Missing post-exit window'}
    foreach($r in $settled){
        if(!$r.on_ground -or $r.self[0] -lt 0 -or !$r.teammate -or [math]::Abs($r.self[2]-$r.teammate[2]) -gt 32){throw 'No stable landing clear of platform'}
    }
    [pscustomobject]@{accepted=$true;completion_frame=$done.frame;health_loss=0;settled_frames=21;scope='prepared_base2_exit_without_crushing'}
}
