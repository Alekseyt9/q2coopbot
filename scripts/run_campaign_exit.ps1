[CmdletBinding()]
param([switch]$Worker,[int]$Port=31240,[string]$OutputRoot='',[string]$Client='',[string]$SourceRuntime='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $SourceRuntime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1
    $OutputRoot=Join-Path $repo ('workspace/artifacts/campaign-exit-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo
    try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        & $using:hostExe -NoProfile -File $using:script -Worker -Port ($using:Port+$_) -OutputRoot $out -Client $using:Client -SourceRuntime $using:SourceRuntime|Out-Host
        $code=$LASTEXITCODE
        $r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json
        if($code){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;timescale=2;parallelism=2;results=$results}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
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
$scene=Get-Content "$PSScriptRoot/scenarios/base1-campaign-exit.json" -Raw|ConvertFrom-Json
$report=@{accepted=$false;reason='not_run';port=$Port;timescale=2};$server=$null;$bot=$null
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N');$instance=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_harness_instance $instance +set sv_test_unlimited_loopback 1 +map $($scene.map)"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='CampaignBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='20s';frame_paced=$true;mode='campaign';next_map=$scene.next_map};output=@{trace_jsonl=$trace};test=@{teleport_map=$scene.map;teleport=$scene.origin}}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $deadline=(Get-Date).AddSeconds(18);$last=$null
    do{
        if(Test-Path $trace){foreach($line in Get-Content $trace -Tail 4){try{$last=$line|ConvertFrom-Json}catch{}}}
        if($last.map -eq $scene.next_map -and $last.campaign.state -eq 'level_completed'){break}
        if($bot.HasExited -or $server.HasExited){throw 'Trial process exited'}
        Start-Sleep -Milliseconds 100
    }while((Get-Date) -lt $deadline)
    if($last.map -ne $scene.next_map -or $last.campaign.state -ne 'level_completed'){throw 'No confirmed native transition'}
    $rows=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $approach=@($rows|Where-Object {$_.map -eq $scene.map -and $_.goal -eq 'reach_level_exit' -and $_.self[0] -lt -1500 -and $_.sent_command.forward -ne 0})
    if(!$approach.Count -or $approach[0].teammate -or $approach[0].spawncount -eq $last.spawncount){throw 'Exit approach/generation proof absent'}
    if($approach[0].campaign.exit.destination -ne 'base2$base1'){throw 'Wrong BSP destination'}
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne 1 -or $log -match 'client command: (map|gamemap) '){throw 'Repeated placement or forced map change'}
    $report.accepted=$true;$report.reason='accepted';$report.approach=$approach[0];$report.completed=$last;$report.trace=$trace;$report.scope=$scene.scope
}catch{$report.reason=$_.Exception.Message}finally{
    foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}
    foreach($p in @($bot,$server)){if($p){$null=$p.WaitForExit(5000)}}
    $report|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
}
if(!$report.accepted){throw $report.reason}
