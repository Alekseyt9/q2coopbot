[CmdletBinding()]
param([switch]$Worker,[int]$Seed=101,[int]$Port=31680,[string]$OutputRoot='',[string]$Client='')
$ErrorActionPreference='Stop';$repo=Split-Path $PSScriptRoot -Parent
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $OutputRoot=Join-Path $repo ('workspace/artifacts/original-activation-'+(Get-Date -Format yyyyMMdd-HHmmss-fff));New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo;try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $exe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $args=@('-NoProfile','-File',$using:script,'-Worker','-Seed',($using:Seed+$_),'-Port',($using:Port+$_),'-OutputRoot',$out,'-Client',$using:Client)
        $child=Start-Process $using:exe -ArgumentList @($args|ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit();$r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json;if($child.ExitCode){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 15|Set-Content (Join-Path $OutputRoot 'report.json')
    "Original activation: $OutputRoot";if(!$accepted){throw 'Original activation rejected'};return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh trial required'}
New-Item -ItemType Directory $OutputRoot|Out-Null
$runtime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base2 -RuntimeRoot (Join-Path $OutputRoot 'runtime')
$server=$null;$bot=$null;$trace=Join-Path $OutputRoot 'bot.jsonl';$report=@{accepted=$false;seed=$Seed;reason='not_run'}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_seed $Seed +map base2"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='ActivationBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='35s';frame_paced=$true;mode='campaign'};test=@{teleport_map='base2';teleport='672,1792,24.125';campaign_goal='768,1792,24.125'};output=@{trace_jsonl=$trace}}|ConvertTo-Json -Depth 6|Set-Content $config
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $deadline=(Get-Date).AddSeconds(30);$last=$null;$unavailableFrame=$null
    do{
        Start-Sleep -Milliseconds 100
        if(Test-Path $trace){foreach($line in @(Get-Content $trace -Tail 3)){ $doc=$null;try{$doc=[System.Text.Json.JsonDocument]::Parse([string]$line);$candidate=$line|ConvertFrom-Json;if($candidate.map){$last=$candidate}}catch{}finally{if($doc){$doc.Dispose()}}}}
        if($last.campaign.state -eq 'test_waypoint_reached'){break}
        if($last.campaign.dependency.state -eq 'activation_timeout'){throw 'activation_timeout'}
        if($last.campaign.dependency.state -eq 'activation_route_unavailable'){
            if($null -eq $unavailableFrame){$unavailableFrame=$last.frame}
            if($last.frame-$unavailableFrame -gt 50){throw 'activation_route_unavailable'}
        }else{$unavailableFrame=$null}
    }while(!$bot.HasExited -and (Get-Date) -lt $deadline)
    if(!$bot.HasExited){Stop-Process -Id $bot.Id;$null=$bot.WaitForExit(5000)}
    if($last.campaign.state -ne 'test_waypoint_reached'){throw 'Original door passage not completed'}
    $rows=@(Get-Content $trace|ForEach-Object {$_|ConvertFrom-Json})
    if($rows|Where-Object {$_.campaign.dependency.state -eq 'activation_timeout'}){throw 'Activation timed out before crossing'}
    $dependency=@($rows|Where-Object {$_.campaign.dependency.door_model -eq 24 -and $_.campaign.dependency.activation.trigger.model -eq 23})
    $contact=@($rows|Where-Object {$_.self[0]+16 -gt 456 -and $_.self[0]-16 -lt 464 -and $_.self[1]+16 -gt 1728 -and $_.self[1]-16 -lt 1848 -and $_.self[2]+32 -gt 0 -and $_.self[2]-24 -lt 56})
    $opened=@($rows|Where-Object {@($_.movers|Where-Object {$_.model -eq 24 -and $_.origin[2] -lt -60}).Count})
    if(!$dependency.Count -or !$contact.Count -or !$opened.Count -or $last.self[0] -lt 736){throw 'Original BSP activation/observed opening/crossing absent'}
    $released=@($rows|Where-Object {$_.frame -gt $dependency[0].frame -and !$_.campaign.dependency -and $_.campaign.state -eq 'approach_test_waypoint'})
    if(!$released.Count -or $released[0].frame -lt $contact[0].frame -or $released[0].frame -lt $opened[0].frame -or $released[0].frame -gt $opened[0].frame+30){throw 'Observed opening did not promptly release activation goal'}
    if($rows|Where-Object {$_.health -le 0 -or $_.teammate -or $_.map -ne 'base2'}){throw 'Unexpected death/teammate/map'}
    $commands=@(Get-Content (Join-Path $OutputRoot 'bot.err')|Select-String 'client command: teleport ')
    if($commands.Count -ne 1 -or $commands[0] -notmatch 'teleport 672 1792 24.125'){throw 'Initial placement proof invalid'}
    if((Get-Content (Join-Path $OutputRoot 'bot.err') -Raw) -match 'client command: (give|kill|map|gamemap) '){throw 'Forced gameplay command'}
    if(@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"}).Count -ne 1){throw 'Seed acknowledgement absent'}
    $report.accepted=$true;$report.reason='accepted';$report.dependency_frames=$dependency.Count;$report.first_contact_frame=$contact[0].frame;$report.first_open_frame=$opened[0].frame;$report.release_frame=$released[0].frame;$report.last=$last;$report.trace=$trace
    $report.fixture=Get-Content (Join-Path $runtime 'elevator-fixture.json') -Raw|ConvertFrom-Json
    $report.entities_sha256=(Get-FileHash (Join-Path $runtime 'baseq2/maps/base2.ent')).Hash
}catch{$report.reason=$_.Exception.Message}
finally{
    foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}}
    if(Test-Path $trace){$report.last=Get-Content $trace -Tail 1|ConvertFrom-Json}
    $report|ConvertTo-Json -Depth 15|Set-Content (Join-Path $OutputRoot 'report.json')
}
if(!$report.accepted){throw $report.reason}
