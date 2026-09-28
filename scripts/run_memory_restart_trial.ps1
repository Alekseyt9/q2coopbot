[CmdletBinding()]
param([int]$Port=30520,[string]$OutputRoot='',[ValidateSet('base2','base3')][string]$Map='base2',[string]$SessionPath='',[switch]$PreparedRuntime)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/memory-restart-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw "UDP port occupied: $Port"}
$source=Join-Path $repo ('workspace/runtime/q2go-elevator-cycle'+$(if($Map -eq 'base2'){'-base2'}))
if(!$PreparedRuntime){& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map $Map | Out-Null}
if(!(Test-Path (Join-Path $source "baseq2/maps/$Map.aas"))){throw "Prepared $Map runtime unavailable"}
$out=if([IO.Path]::IsPathRooted($OutputRoot)){[IO.Path]::GetFullPath($OutputRoot)}else{[IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))}
if(Test-Path $out){throw "Output already exists: $out"}
$runtime=Join-Path $out 'runtime'
New-Item -ItemType Directory (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
Get-ChildItem -LiteralPath $source -File|Where-Object Extension -in @('.exe','.dll')|Copy-Item -Destination $runtime
Copy-Item (Join-Path $source 'baseq2/game.dll') (Join-Path $runtime 'baseq2/game.dll')
foreach($asset in Get-ChildItem (Join-Path $source 'baseq2') -Recurse -File|Where-Object Extension -in @('.pak','.bsp','.aas','.ent')){
    $relative=[IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName)
    $dest=Join-Path (Join-Path $runtime 'baseq2') $relative
    New-Item -ItemType Directory (Split-Path -Parent $dest) -Force|Out-Null
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item -LiteralPath $asset.FullName -Destination $dest}
}
$client=Join-Path $out 'q2coopbot.exe'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
$session=if($SessionPath){(Resolve-Path -LiteralPath $SessionPath).Path}else{Join-Path $repo 'scripts/scenarios/death-point-memory-restart-session.json'}
$memory=Join-Path $out 'travel-memory.json'
$stop=Join-Path $out 'first.stop'
$token=[guid]::NewGuid().ToString('N')
$common=@{server=@{host='127.0.0.1';port=$Port};run=@{duration='120s';frame_paced=$true}}
$actorCfg=@{server=$common.server;client=@{name='TestHuman';game_dir=(Join-Path $runtime 'baseq2')};run=$common.run;output=@{trace_jsonl=(Join-Path $out 'actor.jsonl')};test=@{session=$session;session_role='actor';scenario_result=(Join-Path $out 'completion.json');idle=$true}}
$firstCfg=@{server=$common.server;client=@{name='GoCoopMate';game_dir=(Join-Path $runtime 'baseq2');memory_file=$memory;memory_session=$token};run=$common.run;output=@{trace_jsonl=(Join-Path $out 'first.jsonl');stop_file=$stop};test=@{session=$session;session_role='observer';scenario_result=(Join-Path $out 'completion.json');scenario_tail_frames=5}}
$secondCfg=@{server=$common.server;client=@{name='GoCoopMate';game_dir=(Join-Path $runtime 'baseq2');memory_file=$memory;memory_session=$token};run=$common.run;output=@{trace_jsonl=(Join-Path $out 'second.jsonl')}}
foreach($pair in @(@('actor',$actorCfg),@('first',$firstCfg),@('second',$secondCfg))){$pair[1]|ConvertTo-Json -Depth 8|Set-Content (Join-Path $out ($pair[0]+'-config.json'))}
function Read-Trace([string]$Path){
    if(!(Test-Path $Path)){return @()}
    @(Get-Content -LiteralPath $Path|ForEach-Object {try{$_|ConvertFrom-Json -ErrorAction Stop}catch{}})
}
$server=$null;$actor=$null;$first=$null;$second=$null
$oldRcon=$env:Q2COOPBOT_TEST_RCON
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="+set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set sv_test_trace_client GoCoopMate +set sv_test_start_client GoCoopMate +map $Map"
    $serverLog=Join-Path $out 'server.log'
    $server=Start-Process -FilePath (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -RedirectStandardOutput $serverLog -RedirectStandardError (Join-Path $out 'server.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(35)
    while(!(Test-Path $serverLog) -or !(Select-String -LiteralPath $serverLog -Pattern 'sv_test_start_client ready:' -Quiet)){
        if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}
        Start-Sleep -Milliseconds 100
    }
    $endpoint=@(Get-NetUDPEndpoint -OwningProcess $server.Id)
    if(!($endpoint|Where-Object {$_.LocalPort -eq $Port -and $_.LocalAddress -eq '127.0.0.1'}) -or ($endpoint|Where-Object {$_.LocalAddress -notin @('127.0.0.1','::1')})){throw 'Server not loopback-only'}
    $actor=Start-Process -FilePath $client -ArgumentList "--config `"$(Join-Path $out 'actor-config.json')`"" -RedirectStandardOutput (Join-Path $out 'actor.log') -RedirectStandardError (Join-Path $out 'actor.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(!(Select-String -LiteralPath $serverLog -Pattern 'TestHuman entered the game' -Quiet)){
        if($actor.HasExited -or (Get-Date) -gt $deadline){throw 'Actor startup failed'}
        Start-Sleep -Milliseconds 100
    }
    $first=Start-Process -FilePath $client -ArgumentList "--config `"$(Join-Path $out 'first-config.json')`"" -RedirectStandardOutput (Join-Path $out 'first.log') -RedirectStandardError (Join-Path $out 'first.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(35)
    $witness=$null
    while((Get-Date) -lt $deadline){
        if($first.HasExited -or $server.HasExited -or $actor.HasExited){throw 'First client or server exited early'}
        $witness=@(Read-Trace (Join-Path $out 'first.jsonl')|Where-Object {$_.goal -eq 'regroup_after_respawn' -and $_.health -gt 0 -and !$_.last_teammate}|Select-Object -First 1)
        if($witness.Count){break}
        Start-Sleep -Milliseconds 100
    }
    if(!$witness.Count){throw 'No death-only return before restart'}
    Set-Content -LiteralPath $stop -Value 'stop'
    $first.WaitForExit(5000)|Out-Null
    if(!$first.HasExited){throw 'First bot ignored stop file'}
    $saved=Get-Content $memory -Raw|ConvertFrom-Json
    if(!$saved.death -or $saved.player -or $saved.completed -or $saved.map -ne $Map -or $saved.server -ne "127.0.0.1:$Port|$token") {throw 'Death-only memory was not saved for this server session'}
    $second=Start-Process -FilePath $client -ArgumentList "--config `"$(Join-Path $out 'second-config.json')`"" -RedirectStandardOutput (Join-Path $out 'second.log') -RedirectStandardError (Join-Path $out 'second.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(65)
    $arrived=$null
    while((Get-Date) -lt $deadline){
        if($second.HasExited -or $server.HasExited -or $actor.HasExited){throw 'Second client or server exited early'}
        $rows=@(Read-Trace (Join-Path $out 'second.jsonl'))
        $arrived=@($rows|Where-Object {$_.goal -eq 'wait_for_teammate' -and !$_.teammate -and $_.health -gt 0 -and [math]::Sqrt([math]::Pow($_.self[0]-$saved.death[0],2)+[math]::Pow($_.self[1]-$saved.death[1],2)) -le 64}|Select-Object -First 1)
        if($arrived.Count){break}
        Start-Sleep -Milliseconds 100
    }
    $firstRows=@(Read-Trace (Join-Path $out 'first.jsonl'))
    $rows=@(Read-Trace (Join-Path $out 'second.jsonl'))
    $return=@($rows|Where-Object {$_.goal -eq 'regroup_after_respawn' -and !$_.teammate -and !$_.last_teammate -and $_.health -gt 0})
    if(!$return.Count -or !$arrived.Count){throw 'Restarted bot did not return to death point'}
    if($return[0].spawncount -ne $saved.generation -or $return[0].frame -le $saved.frame){throw 'Memory restored across wrong generation/frame'}
    foreach($axis in 0..2){if([math]::Abs($return[0].goal_point[$axis]-$saved.death[$axis]) -gt .125){throw 'Restored wrong target'}}
    $distance=[math]::Sqrt([math]::Pow($return[0].self[0]-$saved.death[0],2)+[math]::Pow($return[0].self[1]-$saved.death[1],2))
    $maxFrames=if($Map -eq 'base3'){800}else{400}
    if($distance -le 640 -or $arrived[0].frame-$return[0].frame -gt $maxFrames -or $firstRows[-1].frame -ge $rows[0].frame){throw 'Restart identity/distance/deadline failed'}
    $routeMetrics=& "$PSScriptRoot/measure_return_trace.ps1" -TracePath (Join-Path $out 'second.jsonl') -StartFrame $return[0].frame -EndFrame $arrived[0].frame
    @{accepted=$true;server_pid=$server.Id;first_pid=$first.Id;second_pid=$second.Id;server_session=$saved.server;generation=$saved.generation;death=$saved.death;saved_frame=$saved.frame;restart_frame=$rows[0].frame;return_frame=$return[0].frame;arrival_frame=$arrived[0].frame;initial_distance=$distance;route_metrics=$routeMetrics;first_trace='first.jsonl';second_trace='second.jsonl'}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'report.json')
    Write-Output "PASS: $out"
}finally{
    $env:Q2COOPBOT_TEST_RCON=$oldRcon
    foreach($process in @($second,$first,$actor,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id -ErrorAction SilentlyContinue}}
}
