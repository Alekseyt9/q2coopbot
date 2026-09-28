function Assert-ReturnLip($Rows) {
    $rows=@($Rows | Where-Object { $_.map -eq 'base3' -and $_.frame })
    if(!@($rows | Where-Object { [math]::Abs($_.self[0]-1518) -lt 1 -and [math]::Abs($_.self[1]-1400.25) -lt 1 -and $_.on_ground }).Count){throw 'Start position missing'}
    $takeoff=@($rows | Where-Object {$_.arbitration.limit_reason -eq 'jump_takeoff'})
    $land=@($rows | Where-Object {$_.arbitration.limit_reason -eq 'jump_landed' -and $_.on_ground -and $_.self[0] -lt 1440 -and [math]::Abs($_.self[2]+839.875) -lt 2})
    if($takeoff.Count -ne 1 -or $land.Count -ne 1 -or $land[0].frame -le $takeoff[0].frame){throw 'Missing single confirmed crossing'}
    if(@($rows | Where-Object {$_.arbitration.limit_reason -in @('jump_missed','jump_aborted','runup_lost_ground')}).Count){throw 'Jump failed'}
    if($rows[-1].goal -ne 'cover_teammate' -or $rows[-1].health -lt 97){throw 'No safe arrival'}
    [pscustomobject]@{takeoff=$takeoff[0].frame;landing=$land[0].frame;position=$land[0].self;final=$rows[-1].self}
}
