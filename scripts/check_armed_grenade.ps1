function Assert-ArmedGrenade([object[]]$Rows,[object[]]$ActorRows,[switch]$Contact) {
    $priming=@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'test_grenade_arming' -and ($_.sent_command.Buttons -band 1)})
    if($priming.Count -lt 5 -or $priming.Count -gt 20){throw 'Native grenade priming not bounded/proven'}
    if(@($priming|Where-Object gun_frame -EQ 11).Count -lt 2){throw 'Native fuse activation before handoff not proven'}
    $hand=@($Rows|Where-Object {$_.frame -ge $priming[0].frame -and $_.weapon -match '/v_handgr/'})
    $release=@($hand|Where-Object {$_.gun_frame -eq 11 -and $_.arbitration.limit_reason -eq 'hand_grenade_release' -and !($_.sent_command.Buttons -band 1)})
    if(!$release.Count){throw 'Armed native release not observed'}
    if(@($hand|Where-Object {$_.frame -ge $release[0].frame -and ($_.sent_command.Buttons -band 1)}).Count){throw 'Grenade held or restarted after handoff'}
    $firing=@($hand|Where-Object {$_.gun_frame -ge 1 -and $_.gun_frame -le 15 -and $_.frame -ge $release[0].frame})
    if(@($firing|Where-Object {$_.weapon_request}).Count){throw 'Weapon switch requested before grenade recovery'}
    foreach($row in $firing) {
        foreach($pair in @(@('Pitch',0),@('Yaw',1),@('Roll',2))) {
            $actual=([int]$row.sent_command.($pair[0])+[int]$row.delta_angles[$pair[1]]) -band 65535
            if($actual -ne ([int]$row.view_angles[$pair[1]] -band 65535)){throw 'Native throw pose changed during release'}
        }
    }
    if(!@($hand|Where-Object {$_.gun_frame -eq 12}).Count){throw 'Native throw animation missing'}
    $after=@($Rows|Where-Object {$_.frame -ge $release[0].frame})
    if(@($after|Where-Object {$_.health -ne 100 -or $_.test_health_masked -or (!$Contact -and $_.enemies.Count)}).Count){throw 'Damage, masked health or nonempty loss-of-target fixture'}
    if(!@($after|Where-Object {$_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.inventory|Where-Object {$_.name -eq 'Grenades' -and $_.count -eq 4}).Count}).Count){throw 'Real grenade consumption not confirmed'}
    if(!@($after|Where-Object {$_.weapon -eq 'Blaster' -and $_.frame -ge $release[0].frame+40}).Count){throw 'Recovery past grenade fuse not confirmed'}
    if(@($ActorRows|Where-Object {$_.frame -ge $priming[0].frame -and $_.health -ne 100}).Count){throw 'Teammate damaged by grenade'}
    if(!@($ActorRows|Where-Object {$_.frame -ge $release[0].frame+40}).Count){throw 'Teammate observation past fuse missing'}
    $candidate=@($hand|Where-Object {$_.gun_frame -ge 16 -and $_.grenade_prediction.scope -eq 'nominal_center_hand_timer3_diagnostic' -and $_.grenade_prediction.reason -eq 'not_authorized' -and $_.grenade_prediction.samples.Count -eq 9})
    if(!$candidate.Count){throw 'Diagnostic grenade candidate missing'}
    if(!@($release|Where-Object {$_.grenade_prediction.reason -eq 'armed_fuse_unknown' -and !$_.grenade_prediction.samples.Count}).Count){throw 'Unknown armed fuse treated as known'}
    return [pscustomobject]@{priming_frames=$priming.Count;release_frame=$release[0].frame;grenades_remaining=4;bot_health=100;teammate_health=100;diagnostic_samples=$candidate[0].grenade_prediction.samples.Count}
}
