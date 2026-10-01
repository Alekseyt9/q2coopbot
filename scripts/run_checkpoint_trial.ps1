[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config)
$ErrorActionPreference='Stop'
$configPath=(Resolve-Path $Config).Path
$base=Split-Path $configPath -Parent
$cfg=Get-Content $configPath -Raw|ConvertFrom-Json
foreach($key in $cfg.PSObject.Properties.Name){if($key -notin @('runtime_root','client_exe','checkpoint_exe','output_root','port','timescale','barrier','restore_slots','resume','bot_mode','source_checkpoint')){throw "Unknown checkpoint trial field: $key"}}
if($cfg.timescale -ne 2 -or $cfg.port -lt 1024 -or $cfg.port -gt 65534){throw 'Checkpoint trial requires 2x and a valid port'}
if(Get-NetUDPEndpoint -LocalPort $cfg.port -ErrorAction SilentlyContinue){throw 'Checkpoint port occupied'}
function Trial-Path([string]$Path){if([IO.Path]::IsPathRooted($Path)){return $Path};Join-Path $base $Path}
$source=(Resolve-Path (Trial-Path $cfg.runtime_root)).Path
$client=(Resolve-Path (Trial-Path $cfg.client_exe)).Path
$tool=(Resolve-Path (Trial-Path $cfg.checkpoint_exe)).Path
$out=Trial-Path $cfg.output_root
if(Test-Path $out){throw 'Checkpoint trial requires fresh output'}
$runtime=Join-Path $out 'runtime'
$checkpointDir=Join-Path $out 'checkpoint'
if($cfg.source_checkpoint){
    if(!$cfg.barrier -or !$cfg.restore_slots -or !$cfg.resume){throw 'Shared checkpoint branch requires resume/slot binding'}
    $sharedCheckpoint=(Resolve-Path (Trial-Path $cfg.source_checkpoint)).Path
    # Package files must be private writable copies, never hard links.
    foreach($entry in Get-ChildItem $sharedCheckpoint -Recurse){
        if($entry.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Checkpoint package cannot contain reparse points'}
        $dest=Join-Path $checkpointDir ([IO.Path]::GetRelativePath($sharedCheckpoint,$entry.FullName))
        if($entry.PSIsContainer){New-Item -ItemType Directory -Path $dest -Force|Out-Null}else{New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force|Out-Null;Copy-Item -LiteralPath $entry.FullName -Destination $dest}
    }
}
New-Item -ItemType Directory -Path (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
Get-ChildItem $source -File|Where-Object Extension -In '.exe','.dll'|Copy-Item -Destination $runtime
Copy-Item (Join-Path $source 'baseq2/game.dll') (Join-Path $runtime 'baseq2/game.dll')
foreach($asset in Get-ChildItem (Join-Path $source 'baseq2') -File -Recurse|Where-Object Extension -In '.pak','.bsp','.aas','.ent'){
    $rel=[IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName)
    $dest=Join-Path (Join-Path $runtime 'baseq2') $rel
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force|Out-Null
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
$instance=[guid]::NewGuid().ToString('N')
$trace=Join-Path $out 'bot.jsonl'
$botConfig=@{server=@{host='127.0.0.1';port=$cfg.port};client=@{name='CheckpointBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='75s';frame_paced=$true};output=@{trace_jsonl=$trace};test=@{teleport_map='base2';teleport='304,-1888,-81.875';initial_health=38;hold_position=$true}}
if($cfg.barrier){
    $botConfig.test.checkpoint_control=Join-Path $out 'bot-control.json'
    $scene=Join-Path $PSScriptRoot 'scenarios/base2-checkpoint-barrier.json'
    @{server=$botConfig.server;client=@{name='CheckpointActor';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='75s';frame_paced=$true};output=@{trace_jsonl=(Join-Path $out 'actor.jsonl')};test=@{teleport_map='base2';teleport='832,2292,-214';idle=$true;scenario=$scene;checkpoint_control=(Join-Path $out 'actor-control.json')}}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'actor-config.json') -Encoding utf8
}
$botConfig|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'bot-config.json') -Encoding utf8
function Last-Row {
    if(Test-Path $trace){
        $last=$null
        foreach($line in Get-Content $trace -Tail 5){try{$last=$line|ConvertFrom-Json}catch{}}
        return $last
    }
    return $null
}
function Wait-Row([scriptblock]$Condition){
    $deadline=(Get-Date).AddSeconds(20)
    do{$row=Last-Row;if($row -and (& $Condition $row)){return $row};if($server.HasExited -or $bot.HasExited){throw 'Checkpoint server/client exited'};Start-Sleep -Milliseconds 100}while((Get-Date) -lt $deadline)
    throw 'Checkpoint observation timeout'
}
function Checkpoint-Operation([string]$Action){
    $path=Join-Path $out "$Action-config.json"
    $operation=@{version=1;action=$Action;server="127.0.0.1:$($cfg.port)";instance=$instance;runtime_root=$runtime;checkpoint_dir=$checkpointDir;timeout_ms=10000}
    if($cfg.restore_slots -and $Action -eq 'load'){$operation.bind_participants=$true}
    if($cfg.barrier -and $Action -eq 'save'){$anchor=Last-Row;$operation.barrier=@{id=$instance;map='base2';generation=$anchor.spawncount;frame=([int]$anchor.frame+60);controls=@((Join-Path $out 'bot-control.json'),(Join-Path $out 'actor-control.json'))}}
    $operation|ConvertTo-Json -Depth 8|Set-Content $path -Encoding utf8
    $result=& $tool --config $path 2> (Join-Path $out "$Action.err")
    if($LASTEXITCODE){throw "Checkpoint $Action failed; inspect $Action.err"}
    $result|Set-Content (Join-Path $out "$Action-result.json") -Encoding utf8
    return $result|ConvertFrom-Json
}
$server=$null;$bot=$null;$actor=$null;$oldRcon=$env:Q2COOPBOT_TEST_RCON
$report=[ordered]@{accepted=$false;reason='not_run';timescale=2;port=$cfg.port}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $($cfg.port) +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_harness_instance $instance +set sv_test_unlimited_loopback 1 +set sv_test_trace_client CheckpointBot +map base2`$base1"
    if($cfg.source_checkpoint){$args=$args.Replace('+map base2$base1','+map base1')}
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -RedirectStandardOutput (Join-Path $out 'server.log') -RedirectStandardError (Join-Path $out 'server.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    do{Start-Sleep -Milliseconds 100;$eps=@(Get-NetUDPEndpoint -OwningProcess $server.Id -ErrorAction SilentlyContinue);if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!($eps|Where-Object {$_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $cfg.port}))
    if($eps|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback-only'}

    if(!$cfg.source_checkpoint){
    if($cfg.barrier){
        $actor=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'actor-config.json')`"" -RedirectStandardOutput (Join-Path $out 'actor.log') -RedirectStandardError (Join-Path $out 'actor.err') -WindowStyle Hidden -PassThru
        $spawnDeadline=(Get-Date).AddSeconds(8)
        while(!(Test-Path (Join-Path $out 'actor.jsonl'))){if($actor.HasExited -or (Get-Date) -gt $spawnDeadline){throw 'Actor initial signon failed'};Start-Sleep -Milliseconds 100}
    $bot=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'bot-config.json')`"" -RedirectStandardOutput (Join-Path $out 'bot.log') -RedirectStandardError (Join-Path $out 'bot.err') -WindowStyle Hidden -PassThru
        # start_frame=300 is 15 seconds at 2x; allow signon and polling overhead.
        $deadline=(Get-Date).AddSeconds(25)
        do{
            if($actor.HasExited){throw 'Checkpoint actor exited'}
            $actorRow=$null; if(Test-Path (Join-Path $out 'actor.jsonl')){foreach($line in Get-Content (Join-Path $out 'actor.jsonl') -Tail 5){try{$actorRow=$line|ConvertFrom-Json}catch{}}}
            if($actorRow.scenario.state -eq 'running' -and $actorRow.on_ground){break}
            Start-Sleep -Milliseconds 100
        }while((Get-Date) -lt $deadline)
        if($actorRow.scenario.state -ne 'running'){throw 'Checkpoint actor not in reversible step'}
    }
    if(!$cfg.barrier){
    $bot=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'bot-config.json')`"" -RedirectStandardOutput (Join-Path $out 'bot.log') -RedirectStandardError (Join-Path $out 'bot.err') -WindowStyle Hidden -PassThru
    }
    $before=Wait-Row {param($r) $r.map -eq 'base2' -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    $save=Checkpoint-Operation 'save'
    if($save.state -ne 'saved' -or $save.gameplay_verified){throw 'Invalid native save result'}
    }
    if($cfg.barrier){
        if(!$cfg.source_checkpoint -and !$save.barrier_verified){throw 'Coordinated save barrier not verified'}
        $manifest=Get-Content (Join-Path $out 'checkpoint/manifest.json') -Raw|ConvertFrom-Json
        if($manifest.barrier.participants.Count -ne 2){throw 'Two checkpoint sidecars missing'}
        $captures=@(Get-ChildItem (Join-Path $out 'checkpoint/sidecar') -Filter '*.json'|ForEach-Object {Get-Content $_.FullName -Raw|ConvertFrom-Json})
        foreach($capture in $captures){if($capture.frame -ne $manifest.barrier.frame -or $capture.generation -ne $manifest.barrier.generation -or $capture.planner.captured_frame -ne $capture.frame -or $capture.health -le 0){throw 'Inconsistent captured frame/generation'}}
        $actorCapture=@($captures|Where-Object participant -EQ 'CheckpointActor')
        if($actorCapture.Count -ne 1 -or $actorCapture[0].runner.step_id -ne 'checkpoint-wait' -or $actorCapture[0].runner.captured_frame -ne $manifest.barrier.frame){throw 'Runner capture missing'}
        $botCapture=@($captures|Where-Object participant -EQ 'CheckpointBot')[0]
        if(!$cfg.source_checkpoint){
            $before=Wait-Row {param($r) $r.frame -gt $manifest.barrier.frame -and $r.health -eq 38}
            for($axis=0;$axis -lt 3;$axis++){if([math]::Abs($before.self[$axis]-$botCapture.self[$axis]) -gt 0.125){throw 'World advanced inside save barrier'}}
        }
        $report.barrier=$manifest.barrier;$report.captured_participants=$captures
        $report.accepted=$true;$report.reason='accepted';$report.proof='coordinated_capture_save_and_release';$report.before=$before;$report.server_pid=$server.Id;$report.client_pid=$bot.Id
        if($cfg.restore_slots){
            if(!$cfg.source_checkpoint){
            $originalPids=@($server.Id,$bot.Id,$actor.Id)
            # Stop every writer before waiting: sibling processes can inherit
            # redirected output handles and otherwise prevent EOF indefinitely.
            foreach($p in @($actor,$bot,$server)){if(!$p.HasExited){Stop-Process -Id $p.Id}}
            foreach($p in @($actor,$bot,$server)){if(!$p.WaitForExit(5000)){throw 'Original checkpoint process did not stop'}}
            $restartArgs=$args.Replace('+map base2$base1','+map base1')
            $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $restartArgs -WorkingDirectory $runtime -RedirectStandardOutput (Join-Path $out 'restored-server.log') -RedirectStandardError (Join-Path $out 'restored-server.err') -WindowStyle Hidden -PassThru
            $deadline=(Get-Date).AddSeconds(20)
            do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Restored server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $cfg.port -ErrorAction SilentlyContinue))
            }else{$originalPids=@();$report.source_checkpoint=$sharedCheckpoint}
            $load=Checkpoint-Operation 'load'
            if($load.state -ne 'load_acknowledged'){throw 'Restored server load unconfirmed'}
            $trace=Join-Path $out 'restored-bot.jsonl'
            $botConfig.output.trace_jsonl=$trace;$botConfig.test=@{hold_position=$true}
            if($cfg.resume){$botConfig.test.checkpoint_restore=Join-Path $out 'checkpoint';$botConfig.test.checkpoint_mode=$cfg.bot_mode;$botConfig.test.checkpoint_control=Join-Path $out 'restored-bot-control.json'}
            $botConfig|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'restored-bot-config.json') -Encoding utf8
            @{server=$botConfig.server;client=@{name='CheckpointActor';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='40s';frame_paced=$true};output=@{trace_jsonl=(Join-Path $out 'restored-actor.jsonl')};test=@{idle=$true}}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'restored-actor-config.json') -Encoding utf8
            if($cfg.resume){
                $actorConfig=Get-Content (Join-Path $out 'restored-actor-config.json') -Raw|ConvertFrom-Json -AsHashtable
                $actorConfig.test=@{idle=$true;scenario=$scene;checkpoint_restore=(Join-Path $out 'checkpoint');checkpoint_mode='resume';checkpoint_control=(Join-Path $out 'restored-actor-control.json')}
                $actorConfig|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'restored-actor-config.json') -Encoding utf8
            }
            # Capture connected Actor first; restore connects Bot first deliberately.
            $branchClock=[Diagnostics.Stopwatch]::StartNew()
            $bot=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'restored-bot-config.json')`"" -RedirectStandardOutput (Join-Path $out 'restored-bot.log') -RedirectStandardError (Join-Path $out 'restored-bot.err') -WindowStyle Hidden -PassThru
            $restoredBot=Wait-Row {param($r) $r.map -eq 'base2' -and $r.health -eq 38 -and $r.self_entity -eq $botCapture.self_entity -and $r.inventory_known -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1})}
            $actor=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'restored-actor-config.json')`"" -RedirectStandardOutput (Join-Path $out 'restored-actor.log') -RedirectStandardError (Join-Path $out 'restored-actor.err') -WindowStyle Hidden -PassThru
            $deadline=(Get-Date).AddSeconds(20);$restoredActor=$null
            do{
                if(Test-Path (Join-Path $out 'restored-actor.jsonl')){foreach($line in Get-Content (Join-Path $out 'restored-actor.jsonl') -Tail 5){try{$restoredActor=$line|ConvertFrom-Json}catch{}}}
                if($restoredActor.health -eq 100 -and $restoredActor.self_entity -eq $actorCapture[0].self_entity -and $restoredActor.on_ground){break}
                Start-Sleep -Milliseconds 100
            }while((Get-Date) -lt $deadline)
            if($restoredActor.health -ne 100 -or $restoredActor.self_entity -ne $actorCapture[0].self_entity){throw 'Restored actor identity mismatch'}
            foreach($pair in @(@($restoredBot,$botCapture),@($restoredActor,$actorCapture[0]))){for($axis=0;$axis -lt 3;$axis++){if([math]::Abs($pair[0].self[$axis]-$pair[1].self[$axis]) -gt 1){throw 'Participant received wrong native position'}}}
            foreach($path in @('restored-bot.err','restored-actor.err')){if((Get-Content (Join-Path $out $path) -Raw) -match 'client command: (teleport|give health)'){throw 'Native state replaced by test placement'}}
            $report.proof='new_process_native_slots_reverse_connect';$report.original_pids=$originalPids;$report.restored_pids=@($server.Id,$bot.Id,$actor.Id);$report.restored_bot=$restoredBot;$report.restored_actor=$restoredActor
            if($cfg.resume){
                $botReceipt=Get-Content (Join-Path $out 'restored-bot-control.json.restored.json') -Raw|ConvertFrom-Json
                $actorReceipt=Get-Content (Join-Path $out 'restored-actor-control.json.restored.json') -Raw|ConvertFrom-Json
                if($botReceipt.mode -ne $cfg.bot_mode -or $actorReceipt.mode -ne 'resume' -or $actorReceipt.runner_elapsed -ne $actorCapture[0].runner.elapsed_frames){throw 'Resume receipt mismatch'}
                if($cfg.bot_mode -eq 'resume' -and $botReceipt.planner.goal -ne $botCapture.planner.goal){throw 'Saved planner goal not restored'}
                $deadline=(Get-Date).AddSeconds(28)
                do{
                    foreach($line in Get-Content (Join-Path $out 'restored-actor.jsonl') -Tail 5){try{$completedActor=$line|ConvertFrom-Json}catch{}}
                    if($completedActor.scenario.state -eq 'completed'){break}
                    if($actor.HasExited){throw 'Resumed actor exited before completion'}
                    Start-Sleep -Milliseconds 100
                }while((Get-Date) -lt $deadline)
                if($completedActor.scenario.state -ne 'completed'){throw 'Resumed scenario did not complete'}
                $expectedEnd=[int]$actorReceipt.frame+500-[int]$actorReceipt.runner_elapsed
                if($completedActor.scenario.end_frame -ne $expectedEnd){throw 'Remaining runner budget changed'}
                foreach($path in @('restored-bot.err','restored-actor.err')){if((Get-Content (Join-Path $out $path) -Raw) -match 'client command: (teleport|give health)'){throw 'Resume replayed placement'}}
                $report.proof='new_process_resume_and_fresh';$report.bot_mode=$cfg.bot_mode;$report.bot_receipt=$botReceipt;$report.actor_receipt=$actorReceipt;$report.completed_actor=$completedActor;$report.branch_wall_seconds=$branchClock.Elapsed.TotalSeconds
            }
        }
    }
    if(!$cfg.barrier){
    # Deliberately replace the whole game, not just a teleport. No arbitrary
    # console action comes from the trial definition.
    $udp=[Net.Sockets.UdpClient]::new();try{
        $udp.Connect('127.0.0.1',[int]$cfg.port)
        $bytes=[byte[]]@(255,255,255,255)+[Text.Encoding]::ASCII.GetBytes("rcon $env:Q2COOPBOT_TEST_RCON map base1`n`0")
        $null=$udp.Send($bytes,$bytes.Length)
    }finally{$udp.Dispose()}
    $changed=Wait-Row {param($r) $r.map -eq 'base1' -and $r.spawncount -ne $before.spawncount -and $r.health -eq 100 -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and !($r.inventory|Where-Object name -EQ 'Super Shotgun')}
    $load=Checkpoint-Operation 'load'
    if($load.state -ne 'load_acknowledged' -or $load.gameplay_verified){throw 'Load acknowledgement mislabeled as gameplay proof'}
    $after=Wait-Row {param($r) $r.map -eq 'base2' -and $r.spawncount -ne $before.spawncount -and $r.spawncount -ne $changed.spawncount -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    # Loading the same map again must still create a new generation and fresh
    # inventory, rather than being mistaken for a duplicate old observation.
    $reload=Checkpoint-Operation 'load'
    $again=Wait-Row {param($r) $r.map -eq 'base2' -and $r.spawncount -ne $before.spawncount -and $r.spawncount -ne $changed.spawncount -and $r.spawncount -ne $after.spawncount -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    $distance=0;for($i=0;$i -lt 3;$i++){$distance+=[math]::Pow($before.self[$i]-$after.self[$i],2)}
    if([math]::Sqrt($distance) -gt 1){throw 'Saved position not restored'}
    for($i=0;$i -lt 3;$i++){if([math]::Abs($before.self[$i]-$again.self[$i]) -gt 1){throw 'Same-map reload position not restored'}}
    $null=Wait-Row {param($r) $r.spawncount -eq $again.spawncount -and $r.frame -ge $again.frame+5 -and $r.health -eq 38}
    $stable=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}}|Where-Object {$_.spawncount -eq $again.spawncount -and $_.inventory_known -and $_.inventory_age_frames -le 1})
    if($stable.Count -lt 2){throw 'Insufficient fresh restored inventory rows'}
    if(@($stable|Where-Object {$_.health -ne 38 -or !($_.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -or !($_.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}).Count){throw 'Unstable restored health/inventory'}
    $log=Get-Content (Join-Path $out 'bot.err') -Raw
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne 1 -or @([regex]::Matches($log,'client command: give health ')).Count -ne 1){throw 'Test placement/health reapplied after load'}
    $report.accepted=$true;$report.reason='accepted';$report.before=$before;$report.changed=$changed;$report.after=$after;$report.same_map_reload=$again;$report.fresh_inventory_rows=$stable.Count;$report.position_error=[math]::Sqrt($distance);$report.server_pid=$server.Id;$report.client_pid=$bot.Id
    }
}catch{$report.accepted=$false;$report.reason=$_.Exception.Message}finally{
    foreach($p in @($actor,$bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}
    foreach($p in @($actor,$bot,$server)){if($p){$null=$p.WaitForExit(5000)}}
    $env:Q2COOPBOT_TEST_RCON=$oldRcon
    $report|ConvertTo-Json -Depth 30|Set-Content (Join-Path $out 'report.json') -Encoding utf8
}
if(!$report.accepted){throw "Checkpoint rejected: $($report.reason); inspect $out"}
Write-Output "Checkpoint accepted: $out"
