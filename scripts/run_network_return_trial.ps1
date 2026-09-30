[CmdletBinding()]
param([int]$Port=32310,[int]$ProxyPort=32320,[string]$OutputRoot='',[switch]$PreparedRuntime)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
. "$PSScriptRoot/check_network_return.ps1"
foreach($p in @($Port,$ProxyPort)){if($p -lt 1024 -or $p -gt 65535 -or (Get-NetUDPEndpoint -LocalPort $p -ErrorAction SilentlyContinue)){throw "Invalid or occupied port: $p"}}
if($Port -eq $ProxyPort){throw 'Server and proxy ports must differ'}
if(!$PreparedRuntime){& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2|Out-Null}
$source=Join-Path $repo 'workspace/runtime/q2go-elevator-cycle-base2'
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/network-return-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$out=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path $out){throw 'Output already exists'}
$runtime=Join-Path $out 'runtime'
New-Item -ItemType Directory "$runtime/baseq2/maps" -Force|Out-Null
Get-ChildItem $source -File|Where-Object Extension -in @('.exe','.dll')|Copy-Item -Destination $runtime
Copy-Item "$source/baseq2/game.dll" "$runtime/baseq2/game.dll"
foreach($asset in Get-ChildItem "$source/baseq2" -Recurse -File|Where-Object Extension -in @('.pak','.bsp','.aas','.ent')){
    $dest=Join-Path "$runtime/baseq2" ([IO.Path]::GetRelativePath("$source/baseq2",$asset.FullName))
    New-Item -ItemType Directory (Split-Path -Parent $dest) -Force|Out-Null
    try{New-Item -ItemType HardLink $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
Push-Location $repo
try{foreach($name in @('q2coopbot','q2netfault')){go build -o "$out/$name.exe" "./cmd/$name";if($LASTEXITCODE){throw 'Build failed'}}}finally{Pop-Location}
Copy-Item "$repo/scripts/scenarios/active-return-process-restart-session.json" "$out/session.json"
$memory="$out/travel-memory.json"
$token=[guid]::NewGuid().ToString('N')
$signal="$out/arm.json"
$stop="$out/bot.stop"
@{listen="127.0.0.1:$ProxyPort";server="127.0.0.1:$Port";after_ms=0;duration_ms=1500;decode_quake=$true;arm_signal_path='arm.json';run_ms=90000;events='network.jsonl'}|ConvertTo-Json|Set-Content "$out/relay.json" -Encoding utf8
foreach($role in @('actor','observer')){
    $cfg=@{server=@{host='127.0.0.1';port=$(if($role -eq 'actor'){$Port}else{$ProxyPort})};client=@{name=$(if($role -eq 'actor'){'TestHuman'}else{'GoCoopMate'});game_dir="$runtime/baseq2"};run=@{duration='90s';frame_paced=$true};output=@{trace_jsonl="$out/$role.jsonl"};test=@{session="$out/session.json";session_role=$role;scenario_result="$out/completion.json";idle=($role -eq 'actor')}}
    if($role -eq 'observer'){$cfg.client.memory_file=$memory;$cfg.client.memory_session=$token;$cfg.output.stop_file=$stop;$cfg.test.scenario_tail_frames=5}
    $cfg|ConvertTo-Json -Depth 8|Set-Content "$out/$role-config.json" -Encoding utf8
}
$fixturePaths=@("$out/q2coopbot.exe","$out/q2netfault.exe","$out/session.json","$out/relay.json","$out/actor-config.json","$out/observer-config.json")+@(Get-ChildItem $runtime -Recurse -File|ForEach-Object FullName)
$records=@(Get-HarnessFileRecords -Root $out -Paths $fixturePaths)
$fingerprint=Get-HarnessFingerprint -Records $records
@{fingerprint=$fingerprint;files=$records}|ConvertTo-Json -Depth 8|Set-Content "$out/manifest.json"
function Read-NetworkTrace($path){if(Test-Path $path){@(Get-Content $path|ForEach-Object {try{$_|ConvertFrom-Json -ErrorAction Stop}catch{}})}else{@()}}
$server=$null;$relay=$null;$actor=$null;$bot=$null
$oldRcon=$env:Q2COOPBOT_TEST_RCON
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="+set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set sv_test_start_client GoCoopMate +set sv_test_trace_client GoCoopMate +map base2"
    $server=Start-Process "$runtime/q2ded.exe" -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -RedirectStandardOutput "$out/server.log" -RedirectStandardError "$out/server.err" -PassThru
    $deadline=(Get-Date).AddSeconds(30)
    while(!(Test-Path "$out/server.log") -or !(Select-String "$out/server.log" -Pattern 'sv_test_start_client ready:' -Quiet)){if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'};Start-Sleep -Milliseconds 100}
    $endpoints=@(Get-NetUDPEndpoint -OwningProcess $server.Id)
    if(!($endpoints|Where-Object {$_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $Port}) -or @($endpoints|Where-Object LocalAddress -notin @('127.0.0.1','::1')).Count){throw 'Server not loopback-only'}
    $actor=Start-Process "$out/q2coopbot.exe" -ArgumentList "--config `"$out/actor-config.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/actor.log" -RedirectStandardError "$out/actor.err" -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(!(Select-String "$out/server.log" -Pattern 'TestHuman entered the game' -Quiet)){if($actor.HasExited -or (Get-Date) -gt $deadline){throw 'Actor failed'};Start-Sleep -Milliseconds 100}
    $relay=Start-Process "$out/q2netfault.exe" -ArgumentList "--config `"$out/relay.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/relay.log" -RedirectStandardError "$out/relay.err" -PassThru
    $deadline=(Get-Date).AddSeconds(10)
    while(!(Select-String "$out/relay.log" -Pattern 'relay ready:' -Quiet)){if($relay.HasExited -or (Get-Date) -gt $deadline){throw 'Relay failed'};Start-Sleep -Milliseconds 100}
    $bot=Start-Process "$out/q2coopbot.exe" -ArgumentList "--config `"$out/observer-config.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/observer.log" -RedirectStandardError "$out/observer.err" -PassThru
    $deadline=(Get-Date).AddSeconds(55)
    $armed=$null
    while((Get-Date) -lt $deadline){
        foreach($p in @($server,$relay,$actor,$bot)){if($p.HasExited){throw 'Network return process exited early'}}
        $rows=@(Read-NetworkTrace "$out/observer.jsonl")
        if(!$armed){
            $active=@($rows|Where-Object {$_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0 -and !$_.teammate -and !$_.last_teammate})
            if($active.Count -ge 20){
                $w=$active[-1]
                $armed=@{map=$w.map;generation=$w.spawncount;frame=$w.frame;phase=0;role='observer'}
                $armed|ConvertTo-Json|Set-Content ($signal+'.tmp') -Encoding utf8
                Move-Item ($signal+'.tmp') $signal
                Copy-Item $memory "$out/memory-before-loss.json"
            }
        }else{
            $tail=@($rows|Select-Object -Last 10)
            if($tail.Count -eq 10 -and !@($tail|Where-Object {$_.goal -ne 'wait_for_teammate' -or $_.frame -le $armed.frame}).Count){break}
        }
        Start-Sleep -Milliseconds 100
    }
    if(!$armed){throw 'No active return to arm loss'}
    Set-Content $stop 'stop'
    if(!$bot.WaitForExit(5000) -or $bot.ExitCode -ne 0){throw 'Bot did not stop cleanly'}
    # Terminating the relay flushes its event writer; no fixture files are changed.
    Stop-Process -Id $relay.Id;$relay.WaitForExit()
    $rows=@(Read-NetworkTrace "$out/observer.jsonl")
    $events=@(Read-NetworkTrace "$out/network.jsonl")
    $final=Get-Content $memory -Raw|ConvertFrom-Json
    $before=Get-Content "$out/memory-before-loss.json" -Raw|ConvertFrom-Json
    if($before.completed -or $before.player -or $before.server -ne "127.0.0.1:$ProxyPort|$token" -or $before.server -ne $final.server -or !$before.death){throw 'Invalid pre-loss memory'}
    foreach($axis in 0..2){if([math]::Abs($before.death[$axis]-$final.death[$axis]) -gt .125){throw 'Death point changed'}}
    $proof=Test-NetworkReturn -Rows $rows -Events $events -Signal $armed -Memory $final
    if(@($rows.connection|Sort-Object -Unique).Count -ne 1 -or !(Select-String "$out/observer.err" -Pattern 'decode_errors=0' -Quiet)){throw 'Short loss changed connection or decode failed'}
    if((Get-HarnessFingerprint -Records @(Get-HarnessFileRecords -Root $out -Paths $fixturePaths)) -ne $fingerprint){throw 'Network fixture changed'}
    $metrics=& "$PSScriptRoot/measure_return_trace.ps1" -TracePath "$out/observer.jsonl" -StartFrame $armed.frame -EndFrame $rows[-1].frame
    @{accepted=$true;server_pid=$server.Id;bot_pid=$bot.Id;relay_pid=$relay.Id;server_session=$final.server;map='base2';timescale=2;blackout_ms=1500;scope='bidirectional_udp_loss_during_death_return_same_connection';proof=$proof;route_metrics=$metrics}|ConvertTo-Json -Depth 8|Set-Content "$out/report.json"
    Write-Output "PASS: $out"
}finally{$env:Q2COOPBOT_TEST_RCON=$oldRcon;foreach($p in @($bot,$actor,$relay,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}}
