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

function Assert-GrenadeMovingFriend([object[]]$Rows,[object[]]$ActorRows) {
    $priming=@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'test_grenade_arming'})
    if($priming.Count){throw 'Observer fixture must not arm'}
    $before=@($Rows|Where-Object {$_.frame -lt 350 -and $_.weapon -match '/v_handgr/'})
    $risk=@($before|Where-Object {
        $_.grenade_prediction.friend_motion.frame -eq $_.frame -and
        $_.grenade_prediction.friend_motion.entity -eq $_.teammate_entity -and $_.teammate_entity -gt 0 -and
        [math]::Abs($_.grenade_prediction.friend_motion.velocity[1]) -gt 10 -and
        @($_.grenade_prediction.samples|Where-Object {$_.risk -eq 'teammate_future_contact' -and $_.friend_contact_seconds -gt 0 -and $_.friend_contact_seconds -le 0.5}).Count
    })
    if($risk.Count -lt 2){throw 'Moving teammate contact hazard not observed'}
    if(@($before|Where-Object {$_.sent_command.Buttons -band 1}).Count){throw 'Grenade armed during teammate crossing'}
    $actor=@($ActorRows|Where-Object {$_.frame -ge 60 -and $_.frame -lt 350})
    if($actor.Count -lt 10){throw 'Actor movement observation missing'}
    if(@($Rows|Where-Object {$_.health -ne 100 -or $_.test_health_masked -or $_.projectiles.Count}).Count){throw 'Observer damage or unexpected projectile'}
    $ys=@($actor|ForEach-Object {$_.self[1]}|Measure-Object -Minimum -Maximum)
    if($ys[0].Maximum-$ys[0].Minimum -lt 40){throw 'Teammate did not physically cross'}
    if(@($actor|Where-Object {$_.health -ne 100 -or $_.test_health_masked}).Count){throw 'Teammate damaged or masked during crossing'}
    return [pscustomobject]@{hazard_frames=$risk.Count;actor_displacement_y=$ys[0].Maximum-$ys[0].Minimum;arming_during_crossing=$false}
}

function Assert-GrenadeMovingTarget([object[]]$Rows,[object[]]$ActorRows) {
    if(@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'test_grenade_arming'}).Count){throw 'Observer fixture must not arm'}
    $before=@($Rows|Where-Object {$_.frame -lt 350 -and $_.weapon -match '/v_handgr/'})
    $risk=@($before|Where-Object {
        $row=$_
        @($row.grenade_prediction.samples|Where-Object {
            $hit=$_.target_contact
            $motion=@($row.grenade_prediction.target_motion|Where-Object {$_.entity -eq $hit.entity -and $_.frame -eq $row.frame -and $_.class -eq 'monster_insane' -and [math]::Abs($_.velocity[1]) -gt 10})
            $target=@($row.enemies|Where-Object {$_.id -eq $hit.entity -and $_.class -eq 'monster_insane'})
            $hit.seconds -gt 0 -and $hit.seconds -le 0.5 -and $motion.Count -and $target.Count -and $_.risk -match '^moving_target_(contact_unproven|self_blast|teammate_blast)$'
        }).Count -and $row.grenade_prediction.reason -eq 'not_authorized'
    })
    if($risk.Count -lt 2){throw 'Moving target early contact hazard not observed'}
    if(@($before|Where-Object {$_.sent_command.Buttons -band 1}).Count){throw 'Grenade armed during target crossing'}
    if(@($Rows|Where-Object {$_.health -ne 100 -or $_.test_health_masked -or $_.projectiles.Count}).Count){throw 'Observer damage, mask or unexpected projectile'}
    if(@($ActorRows|Where-Object {$_.health -ne 100 -or $_.test_health_masked}).Count){throw 'Teammate damage or health mask'}
    $id=$risk[0].grenade_prediction.samples.target_contact.entity|Where-Object {$_ -gt 0}|Select-Object -First 1
    $positions=@($before.enemies|Where-Object {$_.id -eq $id -and $_.class -eq 'monster_insane'}|ForEach-Object {$_.origin[1]})
    $ys=$positions|Measure-Object -Minimum -Maximum
    if($positions.Count -lt 10 -or $ys.Maximum-$ys.Minimum -lt 40){throw 'Native target did not physically move'}
    return [pscustomobject]@{hazard_frames=$risk.Count;target_entity=$id;target_displacement_y=$ys.Maximum-$ys.Minimum;actual_throw=$false;arming_authorized=$false}
}

function Assert-GrenadeEarlyEnvelope([object[]]$Rows,[ValidateSet('teammate','target')][string]$Kind='teammate') {
    $hazards=@($Rows|Where-Object {
        $row=$_;$envelope=$row.grenade_prediction.early_envelope
        $contacts=@($envelope.contacts|Where-Object {
            $contact=$_
            $identity=if($Kind -eq 'teammate'){$contact.entity -eq $row.teammate_entity -and $row.teammate_entity -gt 0}else{@($row.enemies|Where-Object id -EQ $contact.entity).Count -gt 0}
            $bounds=$contact.min.Count -eq 3 -and $contact.max.Count -eq 3
            if($bounds){for($axis=0;$axis -lt 3;$axis++){if($contact.min[$axis] -gt $contact.max[$axis]){$bounds=$false}}}
            $identity -and $bounds -and $contact.kind -eq $Kind -and $contact.seconds -gt 0 -and $contact.seconds -le 0.5 -and ($contact.self_blast -or $contact.teammate_blast)
        })
        $row.frame -lt 350 -and $row.weapon -match '/v_handgr/' -and
        $envelope.frame -eq $row.frame -and $envelope.scope -eq 'free_flight_500ms_reject_only' -and
        $envelope.status -eq 'reject_early_blast' -and $envelope.authorized -eq $false -and $envelope.geometry_certified -eq $false -and
        $envelope.horizon_seconds -eq 0.5 -and $envelope.assumed_axis_speed -eq 400 -and $contacts.Count
    })
    if($hazards.Count -lt 2){throw 'Continuous jitter early-contact rejection envelope not observed'}
    return [pscustomobject]@{hazard_frames=$hazards.Count;kind=$Kind;horizon_seconds=0.5;assumed_axis_speed=400;geometry_certified=$false;authorized=$false}
}

function Assert-GrenadeGeometry([object[]]$Rows,[object[]]$ActorRows,[switch]$Blocked,[switch]$SurfaceContact) {
    $eligible=@($Rows|Where-Object {$_.frame -lt 350 -and $_.weapon -match '/v_handgr/'})
    $proof=@($eligible|Where-Object {
        $envelope=$_.grenade_prediction.early_envelope;$geometry=$envelope.geometry
        $state=if($Blocked){
            $bounce=$geometry.bounce_envelope;$first=$bounce.ranges|Select-Object -First 1;$last=$bounce.ranges|Select-Object -Last 1
            $bounds=$first.min.Count -eq 3 -and $first.max.Count -eq 3 -and $last.min.Count -eq 3 -and $last.max.Count -eq 3
            if($bounds){for($axis=0;$axis -lt 3;$axis++){if($first.min[$axis] -gt $first.max[$axis] -or $last.min[$axis] -gt $first.min[$axis]-416 -or $last.max[$axis] -lt $first.max[$axis]+416){$bounds=$false}}}
            $surfaceValid=$true
            if($SurfaceContact){
                $surface=$geometry.static_surface_contact;$narrow=$geometry.static_surface_bounce_envelope;$seed=$narrow.ranges|Select-Object -First 1
                $surfaceValid=$surface.min.Count -eq 3 -and $surface.max.Count -eq 3 -and $surface.normal.Count -eq 3 -and $seed.min.Count -eq 3 -and $seed.max.Count -eq 3 -and $narrow.ranges.Count -eq 3 -and $narrow.authorized -eq $false -and $narrow.geometry_certified -eq $false -and $narrow.scope -eq 'coarse_post_collision_model_bound' -and $narrow.model_speed_norm_cap -eq 2000 -and $narrow.gravity -eq $_.gravity -and $seed.tick -eq 3
                if($surfaceValid){
                    $surfaceValid=$surface.normal[0] -eq -1 -and $surface.normal[1] -eq 0 -and $surface.normal[2] -eq 0 -and $surface.distance -eq -256 -and [math]::Abs($surface.min[0]-255.96875) -lt 0.001 -and [math]::Abs($surface.max[0]-255.96875) -lt 0.001
                    for($axis=0;$axis -lt 3;$axis++){if($surface.min[$axis] -gt $surface.max[$axis] -or $seed.min[$axis] -ne $surface.min[$axis] -or $seed.max[$axis] -ne $surface.max[$axis]){$surfaceValid=$false}}
                }
            }elseif($geometry.static_surface_contact -or $geometry.static_surface_bounce_envelope){$surfaceValid=$false}
            $expectedX=if($SurfaceContact){132}else{128}
            $geometry.reason -eq 'possible_static_bounce' -and [math]::Abs($geometry.static_clear_seconds-0.2) -lt 1e-8 -and [math]::Abs($geometry.stop_seconds-0.3) -lt 1e-8 -and [math]::Abs($_.self[0]-$expectedX) -lt 1 -and $surfaceValid -and
            $bounce.scope -eq 'coarse_post_collision_model_bound' -and $bounce.model_speed_norm_cap -eq 2000 -and $bounce.gravity -eq $_.gravity -and $bounce.authorized -eq $false -and $bounce.geometry_certified -eq $false -and $bounce.ranges.Count -eq 3 -and $first.tick -eq 3 -and $last.tick -eq 5 -and $bounds
        }else{$geometry.reason -eq 'free_prefix_only' -and $geometry.static_clear_seconds -eq 0.5 -and !$geometry.stop_seconds}
        $geometry.scope -eq 'static_free_flight_prefix_only' -and $geometry.authorized -eq $false -and $geometry.post_bounce_certified -eq $false -and $envelope.frame -eq $_.frame -and $envelope.geometry_certified -eq $false -and $state
    })
    if($proof.Count -lt 2){throw 'Native static geometry prefix evidence missing'}
    if(@($eligible|Where-Object {$_.sent_command.Buttons -band 1 -or $_.arbitration.limit_reason -eq 'test_grenade_arming'}).Count){throw 'Geometry observer armed grenade'}
    if($Rows.Count -lt 10 -or $ActorRows.Count -lt 10 -or @($Rows+$ActorRows|Where-Object {$_.health -ne 100 -or $_.test_health_masked -or $_.projectiles.Count}).Count){throw 'Observer damage, masking or unexpected projectile'}
    return [pscustomobject]@{frames=$proof.Count;blocked=[bool]$Blocked;static_clear_seconds=$proof[0].grenade_prediction.early_envelope.geometry.static_clear_seconds;post_bounce_certified=$false;authorized=$false}
}
