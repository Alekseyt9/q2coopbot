function Assert-GroundPredictionRunIn($Rows){
    $warm=@($Rows|Where-Object {$_.arbitration.move_source -eq 'test_combat_run_in'})
    if($warm.Count -ne 3){throw 'Ground prediction validation requires the3-command run-in fixture'}
    $count=0;$positionError=0.0;$velocityError=0.0
    for($i=0;$i -lt $Rows.Count-1;$i++){
        $r=$Rows[$i];$next=$Rows[$i+1]
        if($r.frame -lt $warm[0].frame -or $r.frame -gt $warm[-1].frame+5){continue}
        if($next.frame -ne $r.frame+1 -or $next.spawncount -ne $r.spawncount -or $next.map -ne $r.map){throw 'Nonconsecutive prediction observations'}
        if(!$r.ground_prediction -or $r.ground_prediction.model -ne 'baseq2_dry_flat_no_contacts'){throw 'Missing flat-ground prediction'}
        if($r.ground_surface -ne 'dry_flat'){throw "Unsupported run-in surface: $($r.ground_surface)"}
        if($r.ground_prediction.command_path -ne 'static_sampled_clear'){throw "Run-in command path not statically clear: $($r.ground_prediction.command_path)"}
        if(!$r.ground_prediction.neutral_path -or !$r.ground_prediction.command_then_stop_path){throw 'Missing stopping-path diagnostics'}
        if(!$r.on_ground -or !$next.on_ground -or [math]::Abs($r.self[2]-$next.self[2]) -gt 0.125){throw 'Run-in flat-ground segment changed'}
        $pe=0.0;$ve=0.0
        foreach($axis in 0..1){
            $pe+=[math]::Pow($next.self[$axis]-$r.self[$axis]-$r.ground_prediction.displacement[$axis],2)
            $ve+=[math]::Pow($next.self_velocity[$axis]-$r.ground_prediction.velocity[$axis],2)
        }
        $positionError=[math]::Max($positionError,[math]::Sqrt($pe))
        $velocityError=[math]::Max($velocityError,[math]::Sqrt($ve))
        $count++
    }
    if($count -ne 8 -or $positionError -gt 0.25 -or $velocityError -gt 0.25){throw "Ground prediction mismatch: frames=$count position=$positionError velocity=$velocityError"}
    [pscustomobject]@{frames=$count;max_position_error=$positionError;max_velocity_error=$velocityError;scope='prepared_open_flat_ground_acceleration_and_reversal'}
}
