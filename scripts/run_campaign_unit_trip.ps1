[CmdletBinding()]
param([switch]$Worker,[switch]$Blocked,[switch]$Checkpoint,[int]$Seed=101,[int]$Port=31670,[string]$OutputRoot='',[string]$Client='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
function Read-LastCompleteSnapshot([string]$Path){
    if(!(Test-Path $Path)){return $null}
    # Reading a file while it grows can deliver several rows or a prefix of
    # the newest row. Require complete JSON and return exactly one snapshot.
    $latest=$null
    foreach($line in @(Get-Content $Path -Tail 3)){
        $document=$null
        try{
            $document=[System.Text.Json.JsonDocument]::Parse([string]$line)
            $candidate=$line|ConvertFrom-Json
            if($candidate.map -and $candidate.campaign){$latest=$candidate}
        }catch{}finally{if($document){$document.Dispose()}}
    }
    return $latest
}
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $records=Get-HarnessSourceRecords $repo;$fingerprint=Get-HarnessFingerprint $records
    $OutputRoot=Join-Path $repo ('workspace/artifacts/campaign-unit-trip-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo
    try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    if($Checkpoint){Push-Location $repo;try{go build -o (Join-Path $OutputRoot 'q2checkpoint.exe') ./cmd/q2checkpoint;if($LASTEXITCODE){throw 'Checkpoint build failed'}}finally{Pop-Location}}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $args=@('-NoProfile','-File',$using:script,'-Worker','-Seed',($using:Seed+$_),'-Port',($using:Port+$_),'-OutputRoot',$out,'-Client',$using:Client)
        if($using:Blocked){$args+='-Blocked'}
        if($using:Checkpoint){$args+='-Checkpoint'}
        $child=Start-Process $using:hostExe -ArgumentList @($args|ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit()
        $r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json
        if($child.ExitCode){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;checkpoint=[bool]$Checkpoint;negative_control=[bool]$Blocked;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 30|Set-Content (Join-Path $OutputRoot 'report.json')
    "Campaign unit trip: $OutputRoot"
    if(!$accepted){throw 'Campaign unit trip rejected'};return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh trial required'}
New-Item -ItemType Directory $OutputRoot|Out-Null
$runtime=& "$PSScriptRoot/prepare_campaign_unit_trip.ps1" -RuntimeRoot (Join-Path $OutputRoot 'runtime') -Blocked:$Blocked
$trace=Join-Path $OutputRoot 'bot.jsonl';$server=$null;$bot=$null
$report=@{accepted=$false;seed=$Seed;timescale=2;port=$Port;reason='not_run'}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $instance=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_harness_instance $instance +set sv_test_unlimited_loopback 1 +set g_test_seed $Seed +map unit_a"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    $cfg=@{server=@{host='127.0.0.1';port=$Port};client=@{name='UnitBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='45s';frame_paced=$true;mode='campaign';campaign_route=@('unit_a','unit_c');campaign_unit_maps=@('unit_b')};output=@{trace_jsonl=$trace}}
    if($Checkpoint){$cfg.test=@{checkpoint_control=(Join-Path $OutputRoot 'control.json');hold_position_map='unit_b'}}
    $cfg|ConvertTo-Json -Depth 6|Set-Content $config
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $originalTrace=$trace
    if($Checkpoint){
        $deadline=(Get-Date).AddSeconds(15);$anchor=$null
        do{
            $anchor=Read-LastCompleteSnapshot $trace
            if($anchor.map -eq 'unit_b' -and $anchor.on_ground -and $anchor.frame -ge 12){break}
            if($bot.HasExited){throw 'Bot stopped before unit checkpoint'}
            Start-Sleep -Milliseconds 30
        }while((Get-Date) -lt $deadline)
        if($anchor.map -ne 'unit_b' -or !$anchor.on_ground -or $anchor.campaign.unit_trip.state -ne 'activate'){throw 'Remote checkpoint anchor absent'}
        $package=Join-Path $OutputRoot 'checkpoint';$tool=Join-Path (Split-Path $OutputRoot -Parent) 'q2checkpoint.exe'
        $operation=@{version=1;action='save';server="127.0.0.1:$Port";instance=$instance;runtime_root=$runtime;checkpoint_dir=$package;timeout_ms=10000;barrier=@{id=$instance;map='unit_b';generation=$anchor.spawncount;frame=([int]$anchor.frame+30);controls=@($cfg.test.checkpoint_control)}}
        $opPath=Join-Path $OutputRoot 'save-config.json';$operation|ConvertTo-Json -Depth 8|Set-Content $opPath
        $null=& $tool --config $opPath 2> (Join-Path $OutputRoot 'save.err');if($LASTEXITCODE){throw 'Unit coordinated save failed'}
        $capture=Get-Content (Join-Path $package 'sidecar/UnitBot.json') -Raw|ConvertFrom-Json
        if($capture.planner.map -ne 'unit_b' -or $capture.planner.campaign.unit_trip.trip.state -ne 'activate' -or $capture.planner.campaign.map -ne 'unit_a'){throw 'Unit stack absent in native save'}
        Stop-Process -Id $bot.Id;$null=$bot.WaitForExit(5000)
        $operation.Remove('barrier');$operation.action='load';$operation.bind_participants=$true;$operation.hold_after_load=$true
        $opPath=Join-Path $OutputRoot 'load-config.json';$operation|ConvertTo-Json -Depth 8|Set-Content $opPath
        $load=& $tool --config $opPath 2> (Join-Path $OutputRoot 'load.err');if($LASTEXITCODE){throw 'Unit native load failed'};$load=$load|ConvertFrom-Json
        $trace=Join-Path $OutputRoot 'restored-bot.jsonl';$control=Join-Path $OutputRoot 'restored-control.json'
        $cfg.output.trace_jsonl=$trace;$cfg.test=@{checkpoint_restore=$package;checkpoint_mode='resume';checkpoint_control=$control}
        $config=Join-Path $OutputRoot 'restored-config.json';$cfg|ConvertTo-Json -Depth 8|Set-Content $config
        $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'restored-bot.log') -RedirectStandardError (Join-Path $OutputRoot 'restored-bot.err')
        $deadline=(Get-Date).AddSeconds(15)
        while(!(Test-Path ($control+'.restored.json'))){if($bot.HasExited -or (Get-Date) -gt $deadline){throw 'Unit restore receipt absent'};Start-Sleep -Milliseconds 50}
        $receipt=Get-Content ($control+'.restored.json') -Raw|ConvertFrom-Json
        if($receipt.planner.campaign.unit_trip.trip.state -ne 'activate' -or $receipt.frame -ne $load.load_anchor.frame -or $receipt.generation -eq $capture.generation -or $receipt.planner.campaign.destination -ne 'unit_c'){throw 'Unit restore identity/objective mismatch'}
        $before=$capture.planner.campaign.unit_trip.trip;$after=$receipt.planner.campaign.unit_trip.trip
        if([int]$before.elapsed_frames -ne [int]$after.elapsed_frames -or ($before.stack.kind -join ',') -ne ($after.stack.kind -join ',') -or $after.action_attempted -or ($after.stack[1].path -join ',') -ne 'unit_b,unit_a' -or [int]$receipt.planner.campaign.route_index -ne 0){throw 'Unit restore refreshed budget or lost stack/return itinerary'}
        $operation.Remove('bind_participants');$operation.Remove('hold_after_load');$operation.action='release';$operation.release=@{id=$load.load_anchor.id;map=$load.load_anchor.map;frame=$load.load_anchor.frame;generation=$load.load_anchor.generation;controls=@($control)}
        $opPath=Join-Path $OutputRoot 'release-config.json';$operation|ConvertTo-Json -Depth 8|Set-Content $opPath
        $null=& $tool --config $opPath 2> (Join-Path $OutputRoot 'release.err');if($LASTEXITCODE){throw 'Unit restore release failed'}
        $report.checkpoint=@{capture=$capture;receipt=$receipt;load=$load;mode='resume';scope='Native save on remote map before touch; new client; fresh activation and effect evidence after load'}
    }
    $deadline=(Get-Date).AddSeconds(40)
    do{
        Start-Sleep -Milliseconds 100
        $complete=Read-LastCompleteSnapshot $trace
        if($complete){$last=$complete}
        if($last.map -eq 'unit_c' -and $last.campaign.state -eq 'campaign_completed'){break}
        if($Blocked -and $last.campaign.unit_trip.state -eq 'effect_unconfirmed'){break}
        if($last.campaign.unit_trip.state -in @('unit_goal_timeout','unexpected_unit_map','effect_unconfirmed')){throw $last.campaign.unit_trip.state}
    }while(!$bot.HasExited -and (Get-Date) -lt $deadline)
    # Freeze the owned writer before validating the entire trace.
    if(!$bot.HasExited){Stop-Process -Id $bot.Id;$null=$bot.WaitForExit(5000)}
    if($Blocked){
        if($last.map -ne 'unit_a' -or $last.campaign.unit_trip.state -ne 'effect_unconfirmed'){throw 'Negative control failed to retain blocked objective'}
    }elseif($last.map -ne 'unit_c' -or $last.campaign.state -ne 'campaign_completed'){throw 'Native unit cycle not completed'}
    $traces=if($Checkpoint){@($originalTrace,$trace)}else{@($trace)}
    $rows=@(Get-Content $traces|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $visits=@();foreach($row in $rows){if(!$visits.Count -or $row.map -ne $visits[-1].map){$visits+=@{map=$row.map;generation=$row.spawncount;frame=$row.frame}}}
    $expected=if($Blocked){'unit_a,unit_b,unit_a'}else{'unit_a,unit_b,unit_a,unit_c'}
    $generations=if($Blocked){3}else{4}
    if(($visits.map -join ',') -ne $expected -or @($visits.generation|Select-Object -Unique).Count -ne $generations){throw 'Native round trip/generations absent'}
    $tripRows=@($rows|Where-Object {$_.campaign.unit_trip})
    if(!$tripRows.Count -or @($tripRows|Where-Object {[int]$_.campaign.route_index -ne 0 -or [int]$_.campaign.completed_levels -ne 0}).Count){throw 'Remote visit advanced campaign'}
    $confirmed=@($tripRows|Where-Object {$_.map -eq 'unit_a' -and $_.campaign.unit_trip.state -eq 'effect_confirmed'})
    if(!@($tripRows|Where-Object {$_.map -eq 'unit_b' -and $_.campaign.unit_trip.action_attempted}).Count -or (!$Blocked -and !$confirmed.Count) -or ($Blocked -and $confirmed.Count)){throw 'Action attempt/effect evidence invalid'}
    if($rows|Where-Object {$_.teammate -or $_.health -le 0}){throw 'Unexpected teammate/death'}
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if($Checkpoint){$log+=Get-Content (Join-Path $OutputRoot 'restored-bot.err') -Raw}
    if($log -match 'client command: (teleport|give|kill|map|gamemap) '){throw 'Forced gameplay command'}
    $seedAck=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"})
    if($seedAck.Count -ne 1){throw 'Native seed acknowledgement absent'}
    $report.accepted=$true;$report.reason='accepted';$report.seed_verified=$true;$report.visits=$visits;$report.completed=$last.campaign;$report.trace=$trace
    $report.negative_control=[bool]$Blocked
    $report.fixture=Get-Content (Join-Path $runtime 'unit-fixture.json') -Raw|ConvertFrom-Json
    $report.fixture_assets=@(Get-ChildItem (Join-Path $runtime 'baseq2/maps') -Filter 'unit_*'|ForEach-Object {@{name=$_.Name;sha256=(Get-FileHash $_.FullName).Hash}})
}catch{$report.reason=$_.Exception.Message}
finally{
    foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}}
    if(Test-Path $trace){$report.last=Get-Content $trace -Tail 1|ConvertFrom-Json}
    $report|ConvertTo-Json -Depth 30|Set-Content (Join-Path $OutputRoot 'report.json')
}
if(!$report.accepted){throw $report.reason}
