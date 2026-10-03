[CmdletBinding()]
param([switch]$Worker,[switch]$FullLevel,[switch]$Combat,[switch]$Checkpoint,[switch]$Continue,[switch]$Chain,[ValidateSet('','available','empty')][string]$Prepare='',[ValidateRange(0,2147483646)][int]$Seed=101,[int]$Port=31240,[string]$OutputRoot='',[string]$Client='',[string]$SourceRuntime='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if($Chain){if($Checkpoint -or $Prepare){throw 'Chain cannot combine checkpoint/preparation fixtures'};$Continue=$true;$FullLevel=$true}
if($Combat){$FullLevel=$true}
if($Checkpoint -and !$Combat){throw 'Campaign checkpoint requires Combat'}
if($Prepare -and ($Combat -or $FullLevel -or $Checkpoint)){throw 'Preparation requires isolated exit fixture'}
if($Continue -and ($Prepare -or ($Combat -and !$Chain) -or ($FullLevel -and !$Chain) -or $Checkpoint)){throw 'Continuation requires navigation fixture or Combat Chain'}
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $SourceRuntime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1 -KeepMonsters:$Combat
    if($Continue){& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2 -KeepMonsters:$Combat | Out-Null}
    $prefix=if($Chain -and $Combat){'campaign-combat-chain-'}elseif($Chain){'campaign-chain-'}elseif($Continue){'campaign-continue-'}elseif($Prepare){'campaign-prepare-'+$Prepare+'-'}elseif($Combat){'campaign-combat-'}elseif($FullLevel){'campaign-level-'}else{'campaign-exit-'}
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
        if($using:Prepare){$workerArgs+=@('-Prepare',$using:Prepare)}
        if($using:Continue){$workerArgs+='-Continue'}
        if($using:Chain){$workerArgs+='-Chain'}
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
    if($asset.Extension -eq '.ent'){Copy-Item $asset.FullName $dest;continue}
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
$sceneName=if($Chain -and $Combat){'base1-base2-campaign-combat-chain.json'}elseif($Chain){'base1-base2-campaign-chain.json'}elseif($Continue){'base1-campaign-continue.json'}elseif($Prepare){'base1-campaign-prepare.json'}elseif($Combat){'base1-campaign-combat.json'}elseif($FullLevel){'base1-campaign-level.json'}else{'base1-campaign-exit.json'}
$scene=Get-Content (Join-Path "$PSScriptRoot/scenarios" $sceneName) -Raw|ConvertFrom-Json
if($Continue){
    $secondRuntime=Join-Path $repo ('workspace/runtime/q2go-elevator-cycle-base2'+$(if($Combat){'-combat'}else{''}))
    foreach($ext in @('aas','ent')){Copy-Item -LiteralPath (Join-Path $secondRuntime "baseq2/maps/base2.$ext") -Destination (Join-Path $runtime "baseq2/maps/base2.$ext") -Force}
}
if($Chain){Copy-Item -LiteralPath (Join-Path $repo 'workspace/runtime/q2go/baseq2/maps/base3.aas') -Destination (Join-Path $runtime 'baseq2/maps/base3.aas') -Force}
if($Prepare){
    $entPath=Join-Path $runtime 'baseq2/maps/base1.ent'
    $blocks=@([regex]::Matches((Get-Content $entPath -Raw),'(?s)\{[^{}]*\}')|ForEach-Object Value)
    $blocks=@($blocks|Where-Object {$_ -notmatch '"classname"\s+"(?:monster_|weapon_|ammo_|item_)'})
    if($Prepare -eq 'available'){
        foreach($item in $scene.items){$blocks+="{`n"+'"classname" "'+$item.class+"`"`n"+'"origin" "'+$item.origin+"`"`n}"}
    }
    [IO.File]::WriteAllText($entPath,($blocks -join "`n"),[Text.Encoding]::ASCII)
}
$report=@{accepted=$false;reason='not_run';port=$Port;timescale=2;seed=$Seed};$server=$null;$bot=$null
if($Prepare){$report.preparation_fixture_sha256=(Get-FileHash $entPath).Hash}
try{
    if($Combat){
        $fixture=Get-Content (Join-Path $SourceRuntime 'elevator-fixture.json') -Raw|ConvertFrom-Json
        if($fixture.map -ne $scene.map -or $fixture.removed_monsters -ne 0 -or $fixture.source_monsters -lt 1 -or $fixture.scope -ne 'original_combat'){throw 'Original combat fixture proof absent'}
        $report.fixture=$fixture
        if($Chain){
            $secondFixture=Get-Content (Join-Path $secondRuntime 'elevator-fixture.json') -Raw|ConvertFrom-Json
            if($secondFixture.map -ne 'base2' -or $secondFixture.removed_monsters -ne 0 -or $secondFixture.source_monsters -lt 1 -or $secondFixture.scope -ne 'original_combat'){throw 'Second-map original combat fixture proof absent'}
            $report.second_fixture=$secondFixture
        }
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
    if($Prepare){$test.initial_health=$scene.initial_health;$test.weapon_switch_fixture='blaster'}
    $duration=if($Chain){'180s'}elseif($Combat){'115s'}elseif($FullLevel){'75s'}elseif($Prepare -or $Continue){'40s'}else{'20s'}
    $run=@{duration=$duration;frame_paced=$true;mode='campaign'}
    if($Continue){$run.campaign_route=$scene.campaign_route}else{$run.next_map=$scene.next_map}
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='CampaignBot';game_dir=(Join-Path $runtime 'baseq2')};run=$run;output=@{trace_jsonl=$trace};test=$test}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
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
    $deadline=(Get-Date).AddSeconds($(if($Chain){175}elseif($Combat){110}elseif($FullLevel){70}elseif($Prepare -or $Continue){36}else{18}));$last=$null;$unreachableStart=$null
    do{
        if(Test-Path $trace){foreach($line in Get-Content $trace -Tail 4){try{$last=$line|ConvertFrom-Json}catch{}}}
        if($last.map -eq $scene.next_map -and $last.campaign.state -eq 'level_completed'){break}
        if($Continue -and !$Chain -and $last.map -eq 'base2' -and $last.frame -ge 40 -and $last.goal -eq 'reach_level_exit' -and $last.campaign.route_index -eq 1){break}
        if($Chain -and $last.map -eq 'base3' -and $last.campaign.state -eq 'campaign_completed'){break}
        if($Chain -and $last.campaign.dependency.state -eq 'activation_route_unavailable'){
            if($null -eq $unreachableStart){$unreachableStart=$last.frame}
            if($last.frame-$unreachableStart -ge 60){throw 'BSP activation discovered but safe route unavailable'}
        }else{$unreachableStart=$null}
        if($bot.HasExited -or $server.HasExited){throw 'Trial process exited'}
        Start-Sleep -Milliseconds 100
    }while((Get-Date) -lt $deadline)
    $finalMap=if($Chain){'base3'}else{$scene.next_map}
    if($last.map -ne $finalMap -or (!$Continue -and $last.campaign.state -ne 'level_completed') -or ($Chain -and $last.campaign.state -ne 'campaign_completed')){throw 'No confirmed native transition'}
    $rows=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $levelRows=@($rows|Where-Object map -EQ $scene.map)
    if($rows|Where-Object teammate){throw 'Solo trial observed a teammate'}
    if($Chain){
        $groups=@($rows|Group-Object map)
        if(($groups.Name -join ',') -ne 'base1,base2,base3' -or $last.campaign.completed_levels -ne 2){throw 'Ordered campaign route proof absent'}
        if(@($groups|ForEach-Object {$_.Group[0].spawncount}|Select-Object -Unique).Count -ne 3){throw 'Distinct native map generations absent'}
    }
    if($Continue){
        $newRows=@($rows|Where-Object map -EQ 'base2')
        $moving=@($newRows|Where-Object {$_.goal -eq 'reach_level_exit' -and $_.campaign.state -eq 'approach_exit' -and $_.campaign.route_index -eq 1 -and $_.campaign.completed_levels -eq 1 -and $_.campaign.exit.destination -like 'base3*' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0)})
        if(!$moving.Count -or $newRows[0].spawncount -eq $rows[0].spawncount){throw 'New-map forward campaign goal absent'}
        $displacement=0
        foreach($row in $moving){$dx=$row.self[0]-$newRows[0].self[0];$dy=$row.self[1]-$newRows[0].self[1];$displacement=[math]::Max($displacement,[math]::Sqrt($dx*$dx+$dy*$dy))}
        if($displacement -lt 64){throw 'New-map native displacement absent'}
        if($newRows|Where-Object {$_.campaign.preparation.map -eq 'base1'}){throw 'Old-map preparation leaked'}
        $report.continuation=@{first=$newRows[0];movement=$moving[-1];displacement=$displacement;scope=$scene.scope;destination='base3'}
    }
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
        $origin=@($scene.origin.Split(',')|ForEach-Object {[double]::Parse($_,[cultureinfo]::InvariantCulture)})
        $start=@($rows|Where-Object {$_.map -eq $scene.map -and [math]::Abs($_.self[0]-$origin[0]) -lt 1 -and [math]::Abs($_.self[1]-$origin[1]) -lt 1}|Select-Object -First 1)
        if(!$start.Count -or $start[0].self[0]-16 -le -1712){throw 'Fixture starts inside exit'}
    }
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if($Checkpoint){$log+=Get-Content (Join-Path $OutputRoot 'restored-bot.err') -Raw}
    $seedAck=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"})
    if($seedAck.Count -ne 1){throw 'Native game seed acknowledgement absent'}
    $report.seed_verified=$true
    $expectedTeleports=if($FullLevel){0}else{1}
    $forbidden=if($Prepare){'client command: (map|gamemap|kill) '}else{'client command: (map|gamemap|give|kill) '}
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne $expectedTeleports -or $log -match $forbidden){throw 'Placement or forced gameplay command'}
    if($Prepare){
        if(@([regex]::Matches($log,'client command: give health 60')).Count -ne 1 -or @([regex]::Matches($log,'client command: give ')).Count -ne 1){throw 'Initial health fixture mismatch'}
        $stages=@($levelRows|Where-Object {$_.campaign.preparation})
        if(!$stages.Count -or $start[0].health -ne 60 -or ($stages|Where-Object {$_.campaign.preparation.budget_frames -ne 200 -or $_.campaign.preparation.spent_frames -gt 200})){throw 'Preparation clock/loadout proof absent'}
        if($Prepare -eq 'available'){
            $ready=$stages|Where-Object {$_.campaign.preparation.state -eq 'ready' -and $_.health -ge 75 -and $_.armor -ge 25 -and ($_.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -ge 10})}|Select-Object -First 1
            $picked=@($levelRows|Where-Object {$_.pickup.state -eq 'confirmed'}|Group-Object {$_.pickup.entity.ToString()+':'+$_.pickup.end_frame}|ForEach-Object {$_.Group[0].pickup})
            if(!$ready -or !($picked|Where-Object class -EQ 'item_armor_jacket') -or !($picked|Where-Object class -EQ 'ammo_shells')){throw 'Combined health/armor/ammo readiness proof absent'}
            $report.preparation_ready=$ready;$report.pickup_confirmations=$picked
        }else{
            if($levelRows|Where-Object {$_.health -ne 60 -or $_.armor -ne 0 -or $_.pickups -or $_.campaign.preparation.state -eq 'ready'}){throw 'Empty fixture acquired resources or claimed readiness'}
            $continued=$stages|Where-Object {$_.campaign.preparation.state -eq 'no_safe_resources' -and $_.goal -eq 'reach_level_exit' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0)}|Select-Object -First 1
            if(!$continued){throw 'Missing resources delayed exit'}
            $report.continued_without_resources=$continued
        }
        $report.preparation_case=$Prepare
    }
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
        $deferred=@($levelRows|Where-Object {$_.resource_risk.state -eq 'deferred' -and $_.resource_risk.reason -in 'current_exposure','route_exposure' -and ($_.enemies|Where-Object id -EQ $_.resource_risk.enemy)})
        # A completed combat route need not spend an optional pickup. Require
        # either inventory confirmation or observed threat-based deferral;
        # isolated pickup fixtures still require actual inventory delta.
        if($confirmed.Count){
            $resumed=@($levelRows|Where-Object {$_.goal -eq 'reach_level_exit' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0) -and $_.frame -gt $confirmed[0].pickup.end_frame})
            if(!$resumed.Count){throw 'Resumed level objective after pickup proof absent'}
            $report.resumed_after_pickup=$resumed[0];$report.resource_policy_evidence='confirmed_pickup'
        }elseif($deferred.Count){
            $resumed=@($levelRows|Where-Object {$_.goal -eq 'reach_level_exit' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0) -and $_.frame -gt $deferred[0].frame})
            if(!$resumed.Count){throw 'Resumed level objective after resource deferral proof absent'}
            $report.resumed_after_resource_deferral=$resumed[0];$report.resource_policy_evidence='observed_threat_deferral'
        }else{throw 'Neither confirmed pickup nor observed resource threat deferral'}
        $report.pickup_confirmations=@($confirmed|ForEach-Object pickup)
        $report.resource_deferrals=@($deferred|Group-Object {$_.resource_risk.entity.ToString()+':'+$_.resource_risk.reason}|ForEach-Object {$_.Group[0].resource_risk})
        $report.metrics.confirmed_pickups=$confirmed.Count
        $lastAttack=$levelRows|Where-Object {$_.sent_command.buttons -band 1}|Select-Object -Last 1
        $afterCombat=@($levelRows|Where-Object {$_.frame -gt $lastAttack.frame -and $_.goal -in 'reach_level_exit','approach_button','touch_button' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0 -or $_.sent_command.up -gt 0)})
        if(!$afterCombat.Count){throw 'Resumed level objective after combat proof absent'}
        $report.last_attack_frame=$lastAttack.frame;$report.resumed_after_combat=$afterCombat[0]
    }
}catch{$report.accepted=$false;$report.reason=$_.Exception.Message}finally{
    foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}
    foreach($p in @($bot,$server)){if($p){$null=$p.WaitForExit(5000)}}
    if($Chain -and (Test-Path $trace)){
        $observed=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
        $report.levels=@($observed|Group-Object map|ForEach-Object {
            $frames=@($_.Group|Group-Object spawncount,frame|ForEach-Object {$_.Group[0]})
            $loss=0;$gain=0;$deaths=0;$prior=$frames[0]
            foreach($row in $frames){if($row.health -le 0 -and $prior.health -gt 0){$deaths++};if($row.spawncount -eq $prior.spawncount){$delta=$row.health-$prior.health;if($delta -lt 0){$loss-=$delta}else{$gain+=$delta}};$prior=$row}
            @{map=$_.Name;frames=$frames.Count;initial_health=$frames[0].health;final_health=$frames[-1].health;minimum_health=($frames.health|Measure-Object -Minimum).Minimum;initial_armor=$frames[0].armor;final_armor=$frames[-1].armor;final_inventory=$frames[-1].inventory;observed_health_loss=$loss;observed_health_gain=$gain;deaths=$deaths;kills=$(if($Combat -or $_.Name -eq 'base3'){$null}else{0});kill_count_basis=$(if($_.Name -eq 'base3'){'destination only; combat not assessed'}elseif($Combat){'native combat telemetry not yet assessed'}else{'monsters removed in navigation fixture'});last_goal=$frames[-1].goal;last_campaign_state=$frames[-1].campaign.state;last_position=$frames[-1].self}
        })
    }
    if($report.accepted){$report.server_chat=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -like 'CampaignBot: *'})}
    if($Combat -and $report.accepted){
        try{
            . "$PSScriptRoot/read_damage_events.ps1"
            $allEvents=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log'))
            $events=@($allEvents|Where-Object {$_.map -eq $scene.map -and $_.spawncount -eq $levelRows[0].spawncount})
            $selfID=$levelRows[0].self_entity
            if(!$events.Count -or $selfID -lt 1){throw 'Native source-generation damage proof absent'}
            $kills=@($events|Where-Object {$_.attacker -eq $selfID -and $_.target_class -like 'monster_*' -and $_.killed})
            $report.metrics.monsters_killed=$kills.Count;$report.metrics.kill_count_basis='native damage events credited to bot in source generation'
            $report.damage_summary=Measure-BotDamage $events $selfID
            $report.metrics.native_health_damage=$report.damage_summary.received_health_damage
            $report.native_damage_events=$events
            if($Chain){
                foreach($level in @($report.levels|Where-Object map -ne 'base3')){
                    $mapRows=@($rows|Where-Object map -eq $level.map)
                    $generation=$mapRows[0].spawncount;$actor=$mapRows[0].self_entity
                    $mapEvents=@($allEvents|Where-Object {$_.map -eq $level.map -and $_.spawncount -eq $generation})
                    $mapAttacks=@($mapRows|Where-Object {$_.sent_command.buttons -band 1})
                    if(!$mapEvents.Count -or !$mapAttacks.Count -or !@($mapRows|Where-Object enemies).Count){throw "Per-map combat proof absent: $($level.map)"}
                    $summary=Measure-BotDamage $mapEvents $actor
                    $level.kills=@($mapEvents|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*' -and $_.killed}).Count
                    $level.kill_count_basis='native damage events credited to bot in this map generation'
                    $level['native_health_damage']=$summary.received_health_damage
                    $level['damage_summary']=$summary
                    $level['attack_frames']=$mapAttacks.Count
                    $level['spawncount']=$generation
                }
                $report.native_damage_events=$allEvents
            }
            if($Checkpoint){
                if(!$load.rng_restored){throw 'Native RNG restore proof absent'}
                $report.checkpoint.rng_restored=$true
            }
        }catch{$report.accepted=$false;$report.reason=$_.Exception.Message}
    }
    $report|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
}
if(!$report.accepted){throw $report.reason}
