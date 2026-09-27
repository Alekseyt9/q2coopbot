. "$PSScriptRoot/check_projectile_aim.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
function Test-ProjectileNoRegression($Lead,$Baseline) {
    $Baseline.kills -gt 0 -and $Lead.kills -ge $Baseline.kills -and
    $Lead.health_damage -ge $Baseline.health_damage -and
    $null -ne $Lead.kill_after_start_frames -and $null -ne $Baseline.kill_after_start_frames -and
    $Lead.kill_after_start_frames -le $Baseline.kill_after_start_frames -and
    ($null -eq $Baseline.cells_spent -or ($null -ne $Lead.cells_spent -and $Lead.cells_spent -le $Baseline.cells_spent))
}
function Read-CombatStart([string]$LogPath) {
    $matches=@(Select-String -LiteralPath $LogPath -Pattern '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=2 seed=0$')
    if($matches.Count -ne 1){throw 'Single two-client combat release missing'}
    [int]$matches[0].Matches[0].Groups[2].Value
}
function Measure-ProjectileCombat($Rows,$Events,[string]$Mode,[bool]$Baseline,[int]$StartFrame=-1,[string]$Range='near',$Fixture=$null) {
    $ErrorActionPreference='Stop'
    if($StartFrame -lt 0){throw 'Combat release frame required'}
    $targetClass='monster_soldier';$targetOrigin=@(1088,328,-32)
    if($Fixture){$targetClass=$Fixture.target_class;$targetOrigin=$Fixture.target_origin}
    $spawn=@($Rows | Where-Object frame -eq $StartFrame | ForEach-Object {$_.enemies} | Where-Object {$_.class -eq $targetClass -and [math]::Abs($_.origin[0]-$targetOrigin[0]) -lt 1 -and [math]::Abs($_.origin[1]-$targetOrigin[1]) -lt 1 -and [math]::Abs($_.origin[2]-$targetOrigin[2]) -lt 1})
    if($spawn.Count -ne 1){throw 'Comparison target not observed'}
    $id=$spawn[0].id
    $start=@($Rows | Where-Object frame -eq $StartFrame)
    if($start.Count -ne 1 -or $start[0].self_entity -lt 1){throw 'Comparison start missing'}
    $start=$start[0]
    if($Fixture){
        if(@($start.enemies).Count -ne 1){throw 'Projectile scene is not isolated'}
        foreach($pair in @(@($start.self,$Fixture.bot_origin),@($start.teammate,$Fixture.actor_origin))){
            for($axis=0;$axis -lt 3;$axis++){if([math]::Abs($pair[0][$axis]-$pair[1][$axis]) -gt 1){throw 'Fixture actor position mismatch'}}
        }
        if($null -ne $Fixture.light -and @($Rows|Where-Object {$_.sent_command.Light -ne $Fixture.light}).Count){throw 'Fixture light not sent'}
    }
    if(@($Rows|Where-Object {$_.frame -lt $StartFrame+10 -and ($_.sent_command.Buttons -band 1)}).Count){throw 'Fire before common observation window ended'}
    if(@($Events|Where-Object {$_.spawncount -eq $start.spawncount -and $_.frame -lt $StartFrame+10}).Count){throw 'Damage before common observation window ended'}
    $signature=@(foreach($n in 0..9) {
        $r=@($Rows|Where-Object frame -eq ($StartFrame+$n))
        if($r.Count -ne 1){throw 'Incomplete initial trajectory'}
        $e=@($r[0].enemies|Where-Object id -eq $id)
        if($e.Count -ne 1){throw 'Initial target missing'}
        if(($r[0].sent_command.Buttons -band 1) -or !$r[0].on_ground){throw 'Unsafe or unsettled comparison start'}
        [pscustomobject]@{self=$r[0].self;teammate=$r[0].teammate;target=$e[0].origin;solid=$e[0].solid;animation=$e[0].frame}
    })
    $distance2=0.0
    for($i=0;$i -lt 3;$i++){$distance2+=($signature[0].target[$i]-$signature[0].self[$i])*($signature[0].target[$i]-$signature[0].self[$i])}
    $distance=[math]::Sqrt($distance2);$crossing=0;$maxCross=0.0
    for($n=1;$n -lt $signature.Count;$n++){
        $dx=$signature[$n].target[0]-$signature[$n].self[0];$dy=$signature[$n].target[1]-$signature[$n].self[1]
        $vx=10*($signature[$n].target[0]-$signature[$n-1].target[0]);$vy=10*($signature[$n].target[1]-$signature[$n-1].target[1])
        $length=[math]::Sqrt($dx*$dx+$dy*$dy)
        if($length -lt 1){throw 'Invalid comparison distance'}
        $cross=[math]::Abs($vx*$dy-$vy*$dx)/$length
        $maxCross=[math]::Max($maxCross,$cross)
        if($cross -ge 30){$crossing++}
    }
    if($Range -eq 'far' -and ($distance -lt 400 -or $crossing -lt 2)){throw 'Far crossing setup was not observed'}
    $active=@($Rows|Where-Object {if($Mode -eq 'projectile_hyper'){$_.weapon -like '*/v_hyperb/*'}else{$_.weapon -eq 'Blaster'}})
    $fire=@($active|Where-Object {$_.arbitration.aim_entity -eq $id -and ($_.sent_command.Buttons -band 1)})
    if($fire.Count -lt 1 -or @($Rows|Where-Object weapon_request).Count){throw 'Fixed weapon comparison not exercised'}
    $led=@($fire|Where-Object {$_.arbitration.lead_seconds -gt 0})
    if(!$Baseline){
        # Outcome comparisons may reject every unsafe forecast. Actual lead
        # exercise is a separate gate in run_projectile_aim_trial.ps1.
        if($led.Count){$null=Assert-ProjectileAim $Rows $Mode $id 1 $targetClass}
    }else{
        if(@($Rows|Where-Object {$_.arbitration.lead_seconds -gt 0}).Count){throw 'Baseline uses lead'}
    }
    foreach($r in @($fire|Where-Object {$_.arbitration.lead_seconds -le 0})){
        $e=@($r.enemies|Where-Object id -eq $id)[0]
        if(!$e){throw 'Fallback target missing'}
        $top=8*(($e.solid -shr 10) -band 63)-32
        $bottom=-8*(($e.solid -shr 5) -band 31)
        $point=@($e.origin[0],$e.origin[1],($e.origin[2]+[math]::Min([math]::Max(22,$bottom+8),$top-8)))
        for($i=0;$i -lt 3;$i++){if([math]::Abs($r.arbitration.aim_point[$i]-$point[$i]) -gt 0.02){throw 'Fallback did not aim at current body'}}
    }
    $mod=if($Mode -eq 'projectile_hyper'){10}else{1}
    $contacts=@($Events|Where-Object {$_.spawncount -eq $start.spawncount -and $_.map -eq 'base1' -and $_.frame -ge $start.frame -and $_.frame -le $Rows[-1].frame -and $_.target -eq $id})
    if(@($contacts|Where-Object {$_.attacker -ne $start.self_entity -or $_.mod -ne $mod}).Count){throw 'Target damaged by another actor or weapon'}
    $hits=@($contacts|Where-Object {$_.live_health_damage -gt 0})
    $kills=@($hits|Where-Object killed)
    $spent=$null
    if($mod -eq 10){
        if($active[0].ammo -ne 100){throw 'Initial Cells missing'}
        $spent=100-$active[-1].ammo
        if($spent -le 0){throw 'HyperBlaster never consumed Cells'}
    }
    [pscustomobject]@{target=$id;mode=$Mode;baseline=$Baseline;range=$Range;initial_distance=$distance;crossing_frames=$crossing;max_crossing_speed=$maxCross;start_frame=$StartFrame;initial_trajectory=($signature|ConvertTo-Json -Compress -Depth 5)
        health_damage=[int](($hits|Measure-Object live_health_damage -Sum).Sum);damage_events=$hits.Count;kills=$kills.Count
        kill_after_start_frames=$(if($kills.Count){$kills[0].frame-$start.frame}else{$null});cells_spent=$spent
        fire_commands=$fire.Count;lead_fire_commands=$led.Count;target_class=$targetClass;scope='fixed_weapon_native_ai_prepared_scene'}
}
