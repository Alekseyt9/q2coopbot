[CmdletBinding()]
param([switch]$Worker,[switch]$FullLevel,[switch]$Combat,[switch]$Checkpoint,[ValidateRange(0,2147483646)][int]$Seed=101,[int]$Port=31240,[string]$OutputRoot='',[string]$Client='',[string]$SourceRuntime='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if($Combat){$FullLevel=$true}
if($Checkpoint -and !$Combat){throw 'Campaign checkpoint requires Combat'}
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $SourceRuntime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1 -KeepMonsters:$Combat
    $prefix=if($Combat){'campaign-combat-'}elseif($FullLevel){'campaign-level-'}else{'campaign-exit-'}
    $OutputRoot=Join-Path $repo ('workspace/artifacts/'+$prefix+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo
    try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    if($Checkpoint){Push-Location $repo;try{go build -o (Join-Path $OutputRoot 'q2checkpoint.exe') ./cmd/q2checkpoint;if($LASTEXITCODE){throw 'Checkpoint build failed'}}finally{Pop-Location}}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $code=1
        $workerArgs=@('-NoProfile','-File',$using:script,'-Worker','-Port',($using:Port+$_),'-Seed',($using:Seed+$_),'-OutputRoot',$out,'-Client',$using:Client,'-SourceRuntime',$using:SourceRuntime)
        if($using:FullLevel){$workerArgs+='-FullLevel'}
        if($using:Combat){$workerArgs+='-Combat'}
        if($using:Checkpoint){$workerArgs+='-Checkpoint'}
        $workerLog=Join-Path $using:OutputRoot "worker-$_.log"
        $workerErr=Join-Path $using:OutputRoot "worker-$_.err"
        $quotedArgs=@($workerArgs|ForEach-Object {'"'+$_+'"'})
        $child=Start-Process $using:hostExe -ArgumentList $quotedArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr
        $child.WaitForExit();$code=$child.ExitCode
        $r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json
        if($code){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
    Write-Output "Campaign exit: $OutputRoot"
    if(!$accepted){throw 'Campaign exit rejected'}
    return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh trial output required'}
$runtime=Join-Path $OutputRoot 'runtime';$trace=Join-Path $OutputRoot 'bot.jsonl'
New-Item -ItemType Directory (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
Get-ChildItem $SourceRuntime -File|Where-Object Extension -In '.exe','.dll'|Copy-Item -Destination $runtime
Copy-Item (Join-Path $SourceRuntime 'baseq2/game.dll') (Join-Path $runtime 'baseq2/game.dll')
foreach($asset in Get-ChildItem (Join-Path $SourceRuntime 'baseq2') -File -Recurse|Where-Object Extension -In '.pak','.aas','.bsp','.ent'){
    $dest=Join-Path (Join-Path $runtime 'baseq2') ([IO.Path]::GetRelativePath((Join-Path $SourceRuntime 'baseq2'),$asset.FullName))
    New-Item -ItemType Directory (Split-Path $dest -Parent) -Force|Out-Null
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
$sceneName=if($Combat){'base1-campaign-combat.json'}elseif($FullLevel){'base1-campaign-level.json'}else{'base1-campaign-exit.json'}
$scene=Get-Content (Join-Path "$PSScriptRoot/scenarios" $sceneName) -Raw|ConvertFrom-Json
$report=@{accepted=$false;reason='not_run';port=$Port;timescale=2;seed=$Seed};$server=$null;$bot=$null
try{
    if($Combat){
        $fixture=Get-Content (Join-Path $SourceRuntime 'elevator-fixture.json') -Raw|ConvertFrom-Json
        if($fixture.map -ne $scene.map -or $fixture.removed_monsters -ne 0 -or $fixture.source_monsters -lt 1 -or $fixture.scope -ne 'original_combat'){throw 'Original combat fixture proof absent'}
        $report.fixture=$fixture
    }
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N');$instance=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_harness_instance $instance +set sv_test_unlimited_loopback 1 +map $($scene.map)"
    if($Combat){$args='+set g_test_damage 1 '+$args}
    $args="+set g_test_seed $Seed "+$args
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    $test=@{}
    if($Checkpoint){$test.checkpoint_control=Join-Path $OutputRoot 'control.json';$test.hold_position=$true}
    if(!$FullLevel){$test=@{teleport_map=$scene.map;teleport=$scene.origin}}
    $duration=if($Combat){'115s'}elseif($FullLevel){'75s'}else{'20s'}
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='CampaignBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration=$duration;frame_paced=$true;mode='campaign';next_map=$scene.next_map};output=@{trace_jsonl=$trace};test=$test}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    if($Checkpoint){
        $deadline=(Get-Date).AddSeconds(10);$anchor=$null
        do{
            if(Test-Path $trace){foreach($line in Get-Content $trace -Tail 4){try{$anchor=$line|ConvertFrom-Json}catch{}}}
            if($anchor.on_ground -and $anchor.frame -ge 12){break}
            Start-Sleep -Milliseconds 20
        }while((Get-Date) -lt $deadline)
        if(!$anchor.on_ground -or $anchor.frame -gt 45){throw 'Safe pre-combat checkpoint window missed'}
        $originalStart=Get-Content $trace -Head 1|ConvertFrom-Json
        $package=Join-Path $OutputRoot 'checkpoint';$tool=Join-Path (Split-Path $OutputRoot -Parent) 'q2checkpoint.exe'
        $operation=@{version=1;action='save';server="127.0.0.1:$Port";instance=$instance;runtime_root=$runtime;checkpoint_dir=$package;timeout_ms=10000;barrier=@{id=$instance;map='base1';generation=$anchor.spawncount;frame=([int]$anchor.frame+30);controls=@($test.checkpoint_control)}}
        $opPath=Join-Path $OutputRoot 'save-config.json';$operation|ConvertTo-Json -Depth 7|Set-Content $opPath -Encoding utf8
        $save=& $tool --config $opPath 2> (Join-Path $OutputRoot 'save.err');if($LASTEXITCODE){throw 'Campaign coordinated save failed'}
        $capture=Get-Content (Join-Path $package 'sidecar/CampaignBot.json') -Raw|ConvertFrom-Json
        if($capture.planner.campaign.destination -ne 'base2'){throw 'Campaign objective absent in checkpoint'}
        Stop-Process -Id $bot.Id;$null=$bot.WaitForExit(5000)
        $operation.Remove('barrier');$operation.action='load';$operation.bind_participants=$true;$operation.hold_after_load=$true
        $opPath=Join-Path $OutputRoot 'load-config.json';$operation|ConvertTo-Json -Depth 7|Set-Content $opPath -Encoding utf8
        $load=& $tool --config $opPath 2> (Join-Path $OutputRoot 'load.err');if($LASTEXITCODE){throw 'Campaign native load failed'};$load=$load|ConvertFrom-Json
        $trace=Join-Path $OutputRoot 'restored-bot.jsonl'
        $restored=Get-Content $config -Raw|ConvertFrom-Json -AsHashtable
        $restored.output.trace_jsonl=$trace;$control=Join-Path $OutputRoot 'restored-control.json'
        $restored.test=@{checkpoint_restore=$package;checkpoint_mode='resume';checkpoint_control=$control}
        $config=Join-Path $OutputRoot 'restored-config.json';$restored|ConvertTo-Json -Depth 7|Set-Content $config -Encoding utf8
        $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'restored-bot.log') -RedirectStandardError (Join-Path $OutputRoot 'restored-bot.err')
        $deadline=(Get-Date).AddSeconds(15)
        while(!(Test-Path ($control+'.restored.json'))){if($bot.HasExited -or (Get-Date) -gt $deadline){throw 'Campaign restore receipt absent'};Start-Sleep -Milliseconds 50}
        $receipt=Get-Content ($control+'.restored.json') -Raw|ConvertFrom-Json
        if($receipt.planner.campaign.destination -ne 'base2' -or $receipt.frame -ne $load.load_anchor.frame -or $receipt.generation -eq $capture.generation){throw 'Campaign restore identity mismatch'}
        $operation.Remove('bind_participants');$operation.Remove('hold_after_load');$operation.action='release'
        $operation.release=@{id=$load.load_anchor.id;map=$load.load_anchor.map;frame=$load.load_anchor.frame;generation=$load.load_anchor.generation;controls=@($control)}
        $opPath=Join-Path $OutputRoot 'release-config.json';$operation|ConvertTo-Json -Depth 7|Set-Content $opPath -Encoding utf8
        $release=& $tool --config $opPath 2> (Join-Path $OutputRoot 'release.err');if($LASTEXITCODE){throw 'Campaign restore release failed'}
        $report.checkpoint=@{capture=$capture;receipt=$receipt;load=$load;release=($release|ConvertFrom-Json);mode='resume';scope='held at native spawn for safe capture; same server, new bot process; no fresh/resume comparison yet'}
    }
    $deadline=(Get-Date).AddSeconds($(if($Combat){110}elseif($FullLevel){70}else{18}));$last=$null
    do{
        if(Test-Path $trace){foreach($line in Get-Content $trace -Tail 4){try{$last=$line|ConvertFrom-Json}catch{}}}
        if($last.map -eq $scene.next_map -and $last.campaign.state -eq 'level_completed'){break}
        if($bot.HasExited -or $server.HasExited){throw 'Trial process exited'}
        Start-Sleep -Milliseconds 100
    }while((Get-Date) -lt $deadline)
    if($last.map -ne $scene.next_map -or $last.campaign.state -ne 'level_completed'){throw 'No confirmed native transition'}
    $rows=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $levelRows=@($rows|Where-Object map -EQ $scene.map)
    if($rows|Where-Object teammate){throw 'Solo trial observed a teammate'}
    if($FullLevel){
        $first=if($Checkpoint){$originalStart}else{$rows[0]}
        if($first.map -ne $scene.map -or [math]::Abs($first.self[0]-128) -gt 1 -or [math]::Abs($first.self[1]+320) -gt 1 -or $first.self[2] -lt 24 -or $first.self[2] -gt 46 -or $first.health -ne 100){throw 'Normal base1 spawn proof absent'}
    }
    $approach=@($rows|Where-Object {$_.map -eq $scene.map -and $_.goal -eq 'reach_level_exit' -and $_.self[0] -lt -1500 -and $_.sent_command.forward -ne 0})
    if(!$approach.Count -or $approach[0].teammate -or $approach[0].spawncount -eq $last.spawncount){throw 'Exit approach/generation proof absent'}
    if($approach[0].campaign.exit.destination -ne 'base2$base1'){throw 'Wrong BSP destination'}
    $button=@($rows|Where-Object {$_.map -eq $scene.map -and $_.goal -eq 'touch_button' -and $_.sent_command.forward -ne 0})
    $opened=@($rows|Where-Object {$_.map -eq $scene.map -and ($_.movers|Where-Object {$_.model -eq 31 -and $_.origin[2] -lt -60})})
    if(!$button.Count -or !$opened.Count){throw 'Button contact/open hatch proof absent'}
    if(!$FullLevel){
        $start=@($rows|Where-Object {$_.map -eq $scene.map -and [math]::Abs($_.self[0]+1632) -lt 1 -and [math]::Abs($_.self[1]-1600) -lt 1}|Select-Object -First 1)
        if(!$start.Count -or $start[0].self[0]-16 -le -1712){throw 'Fixture starts inside exit'}
    }
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if($Checkpoint){$log+=Get-Content (Join-Path $OutputRoot 'restored-bot.err') -Raw}
    $seedAck=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"})
    if($seedAck.Count -ne 1){throw 'Native game seed acknowledgement absent'}
    $report.seed_verified=$true
    $expectedTeleports=if($FullLevel){0}else{1}
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne $expectedTeleports -or $log -match 'client command: (map|gamemap|give|kill) '){throw 'Placement or forced gameplay command'}
    $report.accepted=$true;$report.reason='accepted';$report.approach=$approach[0];$report.button_contact=$button[0];$report.hatch_open=$opened[0];$report.completed=$last;$report.trace=$trace;$report.scope=$scene.scope
    $loss=0;$gain=0;$previous=$levelRows[0];$uniqueFrames=0;$attacks=0
    foreach($row in $levelRows){
        if($row -ne $levelRows[0] -and $row.frame -eq $previous.frame){continue}
        if($row.spawncount -ne $previous.spawncount){throw 'Unexpected respawn/generation during level'}
        $change=$row.health-$previous.health
        if($change -lt 0){$loss-=$change}else{$gain+=$change}
        if($row.sent_command.buttons -band 1){$attacks++}
        $uniqueFrames++;$previous=$row
    }
    $report.start=$levelRows[0]
    $report.metrics=@{initial_health=$levelRows[0].health;final_level_health=$levelRows[-1].health;minimum_health=($levelRows.health|Measure-Object -Minimum).Minimum;observed_health_loss=$loss;observed_health_gain=$gain;attack_frames=$attacks;unique_level_frames=$uniqueFrames;monsters_killed=0;kill_count_basis='monsters removed in navigation fixture';completion_frame=$last.frame}
    $report.chat_commands=@($levelRows|Where-Object chat_message|Select-Object frame,goal,chat_message)
    if($Combat){
        if(!$attacks -or !@($levelRows|Where-Object enemies).Count){throw 'Combat proof absent'}
        $report.metrics.monsters_killed=$null;$report.metrics.kill_count_basis='not available from client observations'
        $confirmed=@($levelRows|Where-Object {$_.pickup.state -eq 'confirmed'}|Group-Object {$_.pickup.entity.ToString()+':'+$_.pickup.end_frame}|ForEach-Object {$_.Group[0]})
        if(!$confirmed.Count){throw 'Confirmed pickup proof absent'}
        $resumed=@($levelRows|Where-Object {$_.goal -eq 'reach_level_exit' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0) -and $_.frame -gt $confirmed[0].pickup.end_frame})
        if(!$confirmed.Count -or !$resumed.Count){throw 'Confirmed pickup and resumed level objective proof absent'}
        $report.pickup_confirmations=@($confirmed|ForEach-Object pickup);$report.resumed_after_pickup=$resumed[0]
        $report.metrics.confirmed_pickups=$confirmed.Count
        $lastAttack=$levelRows|Where-Object {$_.sent_command.buttons -band 1}|Select-Object -Last 1
        $afterCombat=@($levelRows|Where-Object {$_.frame -gt $lastAttack.frame -and $_.goal -in 'reach_level_exit','approach_button','touch_button' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0 -or $_.sent_command.up -gt 0)})
        if(!$afterCombat.Count){throw 'Resumed level objective after combat proof absent'}
        $report.last_attack_frame=$lastAttack.frame;$report.resumed_after_combat=$afterCombat[0]
    }
}catch{$report.accepted=$false;$report.reason=$_.Exception.Message}finally{
    foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}
    foreach($p in @($bot,$server)){if($p){$null=$p.WaitForExit(5000)}}
    if($report.accepted){$report.server_chat=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -like 'CampaignBot: *'})}
    if($Combat -and $report.accepted){
        try{
            . "$PSScriptRoot/read_damage_events.ps1"
            $events=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log')|Where-Object {$_.map -eq $scene.map -and $_.spawncount -eq $levelRows[0].spawncount})
            $selfID=$levelRows[0].self_entity
            if(!$events.Count -or $selfID -lt 1){throw 'Native source-generation damage proof absent'}
            $kills=@($events|Where-Object {$_.attacker -eq $selfID -and $_.target_class -like 'monster_*' -and $_.killed})
            $report.metrics.monsters_killed=$kills.Count;$report.metrics.kill_count_basis='native damage events credited to bot in source generation'
            $report.damage_summary=Measure-BotDamage $events $selfID
            $report.metrics.native_health_damage=$report.damage_summary.received_health_damage
            $report.native_damage_events=$events
            if($Checkpoint){
                if(!$load.rng_restored){throw 'Native RNG restore proof absent'}
                $report.checkpoint.rng_restored=$true
            }
        }catch{$report.accepted=$false;$report.reason=$_.Exception.Message}
    }
    $report|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
}
if(!$report.accepted){throw $report.reason}
