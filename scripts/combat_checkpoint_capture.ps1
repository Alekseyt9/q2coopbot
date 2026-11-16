# Source capture only. Loaded combat continuation is a separate admission gate.
function Start-CombatCheckpointCapture {
    param([string]$Tool,$Bot,[string]$Trace,[string]$OutputRoot,[string]$Runtime,[int]$Port,[int]$Seed,[string]$Map,[int]$Frame)
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    $row=$null
    do{
        if(Test-Path -LiteralPath $Trace){foreach($line in Get-Content -LiteralPath $Trace -Tail 4){try{$candidate=$line|ConvertFrom-Json;if($candidate.spawncount -gt 0){$row=$candidate}}catch{}}}
        if(!$row){if($Bot.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Combat checkpoint signon observation unavailable'};Start-Sleep -Milliseconds 20}
    }while(!$row)
    if($row.frame -ge $Frame){throw 'Combat checkpoint request arrived after root'}
    $configPath=Join-Path $OutputRoot 'combat-save-config.json'
    $request=@{version=1;action='save';server="127.0.0.1:$Port";instance="learning-$Seed";runtime_root=$Runtime;checkpoint_dir=(Join-Path $OutputRoot 'combat-checkpoint');timeout_ms=20000;
        barrier=@{boundary='combat_predecision_v1';id=[guid]::NewGuid().ToString('N');map=$Map;frame=$Frame;generation=$row.spawncount;controls=@((Join-Path $OutputRoot 'combat-control.json'))}}
    $request|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $configPath -Encoding utf8NoBOM
    Start-Process $Tool -ArgumentList @('--config',('"'+$configPath+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'combat-save-result.json') -RedirectStandardError (Join-Path $OutputRoot 'combat-save.err')
}

function Test-CombatCheckpointContinuation {
    param([string]$OutputRoot,[int]$Frame)
    $capture=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-control.json.state.json') -Raw|ConvertFrom-Json
    $release=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-control.json.released.json') -Raw|ConvertFrom-Json
    $result=Get-Content -LiteralPath (Join-Path $OutputRoot 'combat-save-result.json') -Raw|ConvertFrom-Json
    $rows=@(Get-Content -LiteralPath (Join-Path $OutputRoot 'bot.jsonl')|ForEach-Object {$_|ConvertFrom-Json})
    $prefix=@($rows|Where-Object {$_.frame -eq $Frame-1})
    $root=@($rows|Where-Object {$_.frame -eq $Frame})
    $next=@($rows|Where-Object {$_.frame -eq $Frame+1})
    if($capture.error -or $capture.boundary -ne 'combat_predecision_v1' -or $capture.combat.decision_root.frame -ne $Frame -or $capture.combat.observation_memory.anchor.frame -ne $Frame-1 -or $release.id -ne $capture.id -or $release.state -ne 'native_released' -or $result.state -ne 'saved' -or !$result.barrier_verified -or $prefix.Count -ne 1 -or $root.Count -ne 1 -or $next.Count -ne 1){throw 'Combat checkpoint source boundary/continuation unproved'}
    if($root[0].combat_policy.selection.owner -ne 'provider'){throw 'Checkpoint root not controlled by sampled actor'}
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
    @{accepted=$true;proof='native_predecision_source_capture_and_resume';frame=$Frame;prefix_frame=$Frame-1;next_frame=$Frame+1;root_provider=$true;root_native_applied=1;root_native_single_tick=$true;loaded_combat_continuation=$false;fragment_training=$false}
}
