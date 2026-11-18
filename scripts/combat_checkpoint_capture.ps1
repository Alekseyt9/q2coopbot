# Source capture only. Loaded combat continuation is a separate admission gate.
function Start-CombatCheckpointCapture {
    param([string]$Tool,$Bot,[string]$Trace,[string]$OutputRoot,[string]$Runtime,[int]$Port,[int]$Seed,[string]$Map,[int]$Frame,[ValidateSet('','hand_grenade_primed_v1','equipped_weapon_v1','weapon_trigger_phase_v1')][string]$Phase='',[string]$Weapon='')
    if(($Phase -in @('equipped_weapon_v1','weapon_trigger_phase_v1')) -ne [bool]$Weapon){throw 'Observed equipped weapon predicate missing or unexpected'}
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    $row=$null
    do{
        if(Test-Path -LiteralPath $Trace){foreach($line in Get-Content -LiteralPath $Trace -Tail 4){try{$candidate=$line|ConvertFrom-Json;if($candidate.spawncount -gt 0){$row=$candidate}}catch{}}}
        if(!$row){if($Bot.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Combat checkpoint signon observation unavailable'};Start-Sleep -Milliseconds 20}
    }while(!$row)
    if(!$Phase -and $row.frame -ge $Frame){throw 'Combat checkpoint request arrived after root'}
    $captureID=[guid]::NewGuid().ToString('N')
    if($Phase){
        $control=Join-Path $OutputRoot 'combat-control.json'
        $arm=@{version=1;boundary='combat_predecision_v1';phase=$Phase;id=$captureID;map=$Map;frame=$Frame;generation=$row.spawncount}
        if($Weapon){$arm.weapon=$Weapon}
        $arm|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $OutputRoot 'combat-phase-arm.json') -Encoding utf8NoBOM
        $arm|ConvertTo-Json -Depth 8|Set-Content -LiteralPath ($control+'.tmp') -Encoding utf8NoBOM
        Move-Item -LiteralPath ($control+'.tmp') -Destination $control
        $deadline=[DateTime]::UtcNow.AddSeconds(60)
        $capture=$null
        do{
            if(Test-Path -LiteralPath ($control+'.state.json')){try{$capture=Get-Content -LiteralPath ($control+'.state.json') -Raw|ConvertFrom-Json}catch [IO.IOException]{}}
            if(!$capture){if($Bot.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Requested living first-life checkpoint phase not observed'};Start-Sleep -Milliseconds 20}
        }while(!$capture)
        if($capture.error -or $capture.id -ne $captureID -or $capture.frame -lt $Frame -or $capture.phase -or $capture.weapon -or $capture.map -ne $Map -or $capture.generation -ne $row.spawncount){throw 'Observed checkpoint phase did not resolve to a bound exact root'}
        $Frame=[int]$capture.frame
    }
    $configPath=Join-Path $OutputRoot 'combat-save-config.json'
    $request=@{version=1;action='save';server="127.0.0.1:$Port";instance="learning-$Seed";runtime_root=$Runtime;checkpoint_dir=(Join-Path $OutputRoot 'combat-checkpoint');timeout_ms=20000;
        barrier=@{boundary='combat_predecision_v1';id=$captureID;map=$Map;frame=$Frame;generation=$row.spawncount;controls=@((Join-Path $OutputRoot 'combat-control.json'))}}
    $request|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $configPath -Encoding utf8NoBOM
    Start-Process $Tool -ArgumentList @('--config',('"'+$configPath+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'combat-save-result.json') -RedirectStandardError (Join-Path $OutputRoot 'combat-save.err')
}

function Test-CombatCheckpointContinuation {
    param([string]$OutputRoot,[int]$Frame,[ValidateSet('','hand_grenade_primed_v1','equipped_weapon_v1','weapon_trigger_phase_v1')][string]$Phase='',[string]$Weapon='')
    $capture=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-control.json.state.json') -Raw|ConvertFrom-Json
    $release=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-control.json.released.json') -Raw|ConvertFrom-Json
    $result=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-save-result.json') -Raw|ConvertFrom-Json
    $rows=@(Get-Content -LiteralPath (Join-Path $OutputRoot 'bot.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
    $prefix=@($rows|Where-Object {$_.frame -eq $Frame-1})
    $root=@($rows|Where-Object {$_.frame -eq $Frame})
    $next=@($rows|Where-Object {$_.frame -eq $Frame+1})
    if($capture.error -or $capture.boundary -ne 'combat_predecision_v1' -or $capture.combat.decision_root.frame -ne $Frame -or $capture.combat.observation_memory.anchor.frame -ne $Frame-1 -or $release.id -ne $capture.id -or $release.state -ne 'native_released' -or $result.state -ne 'saved' -or !$result.barrier_verified -or $prefix.Count -ne 1 -or $root.Count -ne 1 -or $next.Count -ne 1){throw 'Combat checkpoint source boundary/continuation unproved'}
    if($root[0].combat_policy.selection.owner -ne 'provider'){throw 'Checkpoint root not controlled by sampled actor'}
    if($Phase -eq 'hand_grenade_primed_v1' -and ($root[0].combat_policy.observation.gun_frame -ne 11 -or $root[0].combat_policy.observation.weapon -notin @('Grenades','models/weapons/v_handgr/tris.md2'))){throw 'Captured root does not have requested observed hand-grenade phase'}
    if($Phase -in @('equipped_weapon_v1','weapon_trigger_phase_v1')){
        $models=@{'Blaster'='v_blast';'Shotgun'='v_shotg';'Super Shotgun'='v_shotg2';'Machinegun'='v_machn';'Chaingun'='v_chain';'Grenade Launcher'='v_launch';'Rocket Launcher'='v_rocket';'HyperBlaster'='v_hyperb';'Railgun'='v_rail';'BFG10K'='v_bfg';'Grenades'='v_handgr'}
        if(!$models.ContainsKey($Weapon) -or $root[0].combat_policy.observation.weapon -notin @($Weapon,('models/weapons/'+$models[$Weapon]+'/tris.md2'))){throw 'Captured root does not have requested observed equipped weapon'}
    }
    if($Phase -eq 'weapon_trigger_phase_v1'){
        $o=$root[0].combat_policy.observation;$gf=[int]$o.gun_frame;$ammo=[int]$o.ammo
        $cost=1;if($Weapon -eq 'Blaster'){$cost=0}elseif($Weapon -eq 'Super Shotgun'){$cost=2}elseif($Weapon -eq 'BFG10K'){$cost=50}
        $idle=@{'Blaster'=@(10,52);'Shotgun'=@(20,36);'Super Shotgun'=@(19,57);'Machinegun'=@(7,45);'Chaingun'=@(33,61);'Grenade Launcher'=@(18,59);'Rocket Launcher'=@(14,50);'HyperBlaster'=@(22,49);'Railgun'=@(20,56);'BFG10K'=@(34,55)}
        $ready=($Weapon -eq 'Grenades' -and ($gf -eq 11 -or ($gf -ge 16 -and $gf -le 48))) -or ($Weapon -eq 'Machinegun' -and $gf -in @(4,5)) -or ($Weapon -eq 'HyperBlaster' -and $gf -ge 6 -and $gf -le 11) -or ($Weapon -eq 'Chaingun' -and $gf -in @(14,21))
        if($idle.ContainsKey($Weapon)){$ready=$ready -or ($gf -ge $idle[$Weapon][0] -and $gf -le $idle[$Weapon][1])}
        if(!$ready -or $ammo -lt $cost){throw 'Captured weapon does not have requested observed trigger phase/ammo'}
    }
    $native=@(Get-Content -LiteralPath (Join-Path $OutputRoot 'server.log'))
    $applied=@($native|Where-Object {$_ -match "^sv_test_applied_cmd spawncount=$($capture.generation) frame=$Frame "})
    if($applied.Count -ne 1){throw 'Checkpoint root native application missing or duplicated'}
    $values=@{}
    foreach($kv in [regex]::Matches($applied[0],'([a-z]+)=(-?\d+)')){$values[$kv.Groups[1].Value]=[long]$kv.Groups[2].Value}
    if($values.seq -ne $root[0].client_sequence){throw 'Checkpoint root native sequence mismatch'}
    foreach($key in @('pitch','yaw','roll','forward','side','up','buttons','impulse','msec','light')){
        if($values[$key] -ne $root[0].sent_command.$key){throw "Checkpoint root native command mismatch: $key"}
    }
    $begin=@($native|Where-Object {$_ -match "^sv_test_step version=1 phase=begin spawncount=$($capture.generation) frame=$Frame seq=$($values.seq) actor=$($capture.self_entity)$"})
    $end=@($native|Where-Object {$_ -match "^sv_test_step version=1 phase=end spawncount=$($capture.generation) frame=$($Frame+1) seq=$($values.seq) actor=$($capture.self_entity)$"})
    if($begin.Count -ne 1 -or $end.Count -ne 1){throw 'Checkpoint root native single world tick unproved'}
    @{accepted=$true;proof='native_predecision_source_capture_and_resume';frame=$Frame;phase=$Phase;weapon=$Weapon;prefix_frame=$Frame-1;next_frame=$Frame+1;root_provider=$true;root_native_applied=1;root_native_single_tick=$true;loaded_combat_continuation=$false;fragment_training=$false}
}
