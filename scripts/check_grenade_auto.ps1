function Assert-GrenadeAuto([object[]]$Rows,[object[]]$ActorRows,[switch]$Reject) {
    $active=@($Rows|Where-Object {$_.frame -ge 140})
    if($active.Count -lt 30 -or @($active+$ActorRows|Where-Object {$_.health -ne 100 -or $_.test_health_masked}).Count){throw 'Native automatic grenade damage/masking or observation missing'}
    if(@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'test_grenade_arming'}).Count){throw 'Forced arming used for automatic policy'}
    if(!@($active|Where-Object {$_.enemies.Count -and @($_.enemies|Where-Object clear_shot -EQ $true).Count}).Count){throw 'Native visible target missing'}
    $start=@($active|Where-Object {$_.arbitration.limit_reason -eq 'hand_grenade_auto_start' -and ($_.sent_command.Buttons -band 1)})
    if($Reject){
        if($start.Count -or @($active|Where-Object {$_.projectiles.Count -or $_.weapon_request -eq 'use Grenades' -or ($_.weapon -match '/v_handgr/' -and (($_.sent_command.Buttons -band 1) -or ($_.gun_frame -ge 1 -and $_.gun_frame -le 15)))}).Count){throw 'Unsafe friend fixture started grenade'}
        if(!@($active|Where-Object {$_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.inventory|Where-Object {$_.name -eq 'Grenades' -and $_.count -eq 5}).Count}).Count){throw 'Rejected grenade inventory missing'}
        return @{rejected=$true;grenades_remaining=5;bot_health=100;teammate_health=100}
    }
    if($start.Count -ne 1 -or $start[0].grenade_prediction.samples.Count -ne 9 -or $start[0].grenade_prediction.scope -ne 'sampled_timer2_9_speed413_heuristic'){throw 'Automatic sampled plan missing'}
    if(!@($active|Where-Object weapon_request -EQ 'use Grenades').Count){throw 'Production grenade selection missing'}
    $release=@($active|Where-Object {$_.gun_frame -eq 11 -and $_.arbitration.limit_reason -eq 'hand_grenade_release' -and !($_.sent_command.Buttons -band 1)})
    if(!$release.Count -or $release[0].frame-$start[0].frame -gt 20){throw 'Native automatic release missing/unbounded'}
    $hand=@($active|Where-Object {$_.weapon -match '/v_handgr/' -and $_.frame -gt $start[0].frame})
    if(@($hand|Where-Object {$_.sent_command.Buttons -band 1 -or ($_.gun_frame -ge 1 -and $_.gun_frame -le 15 -and $_.weapon_request)}).Count){throw 'Grenade held/restarted or switched during cycle'}
    if(!@($active|Where-Object gun_frame -EQ 12).Count -or !@($active|Where-Object {$_.projectiles.Count}).Count -or !@($active|Where-Object {$_.explosions.Count}).Count){throw 'Actual native throw/explosion missing'}
    if(!@($active|Where-Object {$_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.inventory|Where-Object {$_.name -eq 'Grenades' -and $_.count -eq 4}).Count}).Count){throw 'Grenade consumption missing'}
    if(!@($active|Where-Object {$_.weapon -eq 'Blaster' -and $_.frame -ge $release[0].frame+40}).Count){throw 'Recovery past fuse missing'}
    return @{start_frame=$start[0].frame;release_frame=$release[0].frame;grenades_remaining=4;bot_health=100;teammate_health=100}
}
