function Assert-ElevatorBoarding($Rows,$Scenario) {
    $scene=@($Rows|Where-Object {$_.map -eq 'base3' -and $_.frame -ge $Scenario.bot_release_frame})
    $last=$null;$outside=$false;$supported=0;$rising=0;$rise=0.0;$wait=0;$board=$false;$ride=$false
    foreach($r in $scene) {
        if($r.frame -gt $Scenario.bot_release_frame+65){break}
        if($r.health -ne 100){throw 'Damage before boarding acceptance'}
        if($last -and ($r.frame -ne $last.frame+1 -or $r.spawncount -ne $last.spawncount)){throw 'Discontinuous boarding trace'}
        $m=@($r.movers|Where-Object model -eq 37)
        if($m.Count -ne 1){throw 'Platform37 observation missing'}
        $inside=$r.self[0] -ge 776 -and $r.self[0] -le 888 -and $r.self[1] -ge 288 -and $r.self[1] -le 392
        if(!$board -and !$inside){$outside=$true}
        if($r.elevator -eq 'wait_bottom' -and $m[0].origin[2] -gt -174){
            $wait++
            if(!$r.on_ground -or [math]::Abs($r.self[2]+423.875) -gt 0.25){throw 'Bot left waiting floor before lift returned'}
        }
        if($r.elevator -eq 'board'){$board=$true}
        if($r.elevator -eq 'ride'){$ride=$true}
        # Bounds include structural geometry above the deck. Correlated server
        # motion and ground contact establish support, not the model's MaxZ.
        $onDeck=$inside -and $r.on_ground -and [math]::Abs(($r.self[2]-$m[0].origin[2])+231.875) -le 0.25
        if($board -and $onDeck){
            $supported++
            if($last -and $last.onDeck -and $last.moverID -eq $m[0].id){
                $dz=$m[0].origin[2]-$last.moverZ
                if($dz -gt 0.125){
                    if([math]::Abs(($r.self[2]-$last.selfZ)-$dz) -gt 0.25){throw 'Bot is not carried by platform'}
                    $rising++;$rise+=$dz
                }
            }
        }
        if($outside -and $board -and $ride -and $supported -ge 4 -and $rising -ge 3 -and $rise -ge 30){
            if($Scenario.name -like '*late' -and $wait -lt 5){throw 'Late arrival did not exercise waiting'}
            return [pscustomobject]@{accepted=$true;boarding_frame=$r.frame;supported_frames=$supported;rising_frames=$rising;carried_height=$rise;waiting_frames=$wait;health_loss=0;scope='prepared_boarding_and_initial_ascent'}
        }
        $last=[pscustomobject]@{frame=$r.frame;spawncount=$r.spawncount;onDeck=$onDeck;moverID=$m[0].id;moverZ=$m[0].origin[2];selfZ=$r.self[2]}
    }
    throw 'No supported boarding and correlated ascent within deadline'
}
