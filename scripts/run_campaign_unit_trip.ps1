[CmdletBinding()]
param([switch]$Worker,[int]$Seed=101,[int]$Port=31670,[string]$OutputRoot='',[string]$Client='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $records=Get-HarnessSourceRecords $repo;$fingerprint=Get-HarnessFingerprint $records
    $OutputRoot=Join-Path $repo ('workspace/artifacts/campaign-unit-trip-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo
    try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $args=@('-NoProfile','-File',$using:script,'-Worker','-Seed',($using:Seed+$_),'-Port',($using:Port+$_),'-OutputRoot',$out,'-Client',$using:Client)
        $child=Start-Process $using:hostExe -ArgumentList @($args|ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit()
        $r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json
        if($child.ExitCode){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 18|Set-Content (Join-Path $OutputRoot 'report.json')
    "Campaign unit trip: $OutputRoot"
    if(!$accepted){throw 'Campaign unit trip rejected'};return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh trial required'}
New-Item -ItemType Directory $OutputRoot|Out-Null
$runtime=& "$PSScriptRoot/prepare_campaign_unit_trip.ps1" -RuntimeRoot (Join-Path $OutputRoot 'runtime')
$trace=Join-Path $OutputRoot 'bot.jsonl';$server=$null;$bot=$null
$report=@{accepted=$false;seed=$Seed;timescale=2;port=$Port;reason='not_run'}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_seed $Seed +map unit_a"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='UnitBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='45s';frame_paced=$true;mode='campaign';campaign_route=@('unit_a','unit_c');campaign_unit_maps=@('unit_b')};output=@{trace_jsonl=$trace}}|ConvertTo-Json -Depth 6|Set-Content $config
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $deadline=(Get-Date).AddSeconds(40)
    do{
        Start-Sleep -Milliseconds 100
        $last=$null
        if(Test-Path $trace){try{$last=Get-Content $trace -Tail 1|ConvertFrom-Json}catch{}}
        if($last.map -eq 'unit_c' -and $last.campaign.state -eq 'campaign_completed'){break}
        if($last.campaign.unit_trip.state -in @('unit_goal_timeout','unexpected_unit_map','effect_unconfirmed')){throw $last.campaign.unit_trip.state}
    }while(!$bot.HasExited -and (Get-Date) -lt $deadline)
    if($last.map -ne 'unit_c' -or $last.campaign.state -ne 'campaign_completed'){throw 'Native unit cycle not completed'}
    $rows=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $visits=@();foreach($row in $rows){if(!$visits.Count -or $row.map -ne $visits[-1].map){$visits+=@{map=$row.map;generation=$row.spawncount;frame=$row.frame}}}
    if(($visits.map -join ',') -ne 'unit_a,unit_b,unit_a,unit_c' -or @($visits.generation|Select-Object -Unique).Count -ne 4){throw 'Native round trip/generations absent'}
    $tripRows=@($rows|Where-Object {$_.campaign.unit_trip})
    if(!$tripRows.Count -or @($tripRows|Where-Object {$_.campaign.route_index -ne 0 -or $_.campaign.completed_levels -ne 0}).Count){throw 'Remote visit advanced campaign'}
    if(!@($tripRows|Where-Object {$_.map -eq 'unit_b' -and $_.campaign.unit_trip.action_attempted}).Count -or !@($tripRows|Where-Object {$_.map -eq 'unit_a' -and $_.campaign.unit_trip.state -eq 'effect_confirmed'}).Count){throw 'Action attempt/observed effect absent'}
    if($rows|Where-Object {$_.teammate -or $_.health -le 0}){throw 'Unexpected teammate/death'}
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if($log -match 'client command: (teleport|give|kill|map|gamemap) '){throw 'Forced gameplay command'}
    $seedAck=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"})
    if($seedAck.Count -ne 1){throw 'Native seed acknowledgement absent'}
    $report.accepted=$true;$report.reason='accepted';$report.seed_verified=$true;$report.visits=$visits;$report.completed=$last.campaign;$report.trace=$trace
    $report.fixture=Get-Content (Join-Path $runtime 'unit-fixture.json') -Raw|ConvertFrom-Json
    $report.fixture_assets=@(Get-ChildItem (Join-Path $runtime 'baseq2/maps') -Filter 'unit_*'|ForEach-Object {@{name=$_.Name;sha256=(Get-FileHash $_.FullName).Hash}})
}catch{$report.reason=$_.Exception.Message}
finally{
    foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}}
    if(Test-Path $trace){$report.last=Get-Content $trace -Tail 1|ConvertFrom-Json}
    $report|ConvertTo-Json -Depth 18|Set-Content (Join-Path $OutputRoot 'report.json')
}
if(!$report.accepted){throw $report.reason}
