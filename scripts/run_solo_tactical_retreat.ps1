[CmdletBinding()]
param([switch]$Worker,[switch]$Cover,[int]$Seed=601,[int]$Port=31820,[string]$OutputRoot='',[string]$Client='',[string]$System1='hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0')
$ErrorActionPreference='Stop';$repo=Split-Path $PSScriptRoot -Parent
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $OutputRoot=Join-Path $repo ('workspace/artifacts/solo-tactical-retreat-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo;try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $args=@('-NoProfile','-File',$using:script,'-Worker','-Seed',($using:Seed+$_),'-Port',($using:Port+$_),'-OutputRoot',$out,'-Client',$using:Client,'-System1',$using:System1)
        if($using:Cover){$args+='-Cover'}
        $child=Start-Process $using:hostExe -ArgumentList @($args|ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit();$r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json;if($child.ExitCode){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;cover=[bool]$Cover;model=$System1;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json')
    "Solo tactical retreat: $OutputRoot";if(!$accepted){throw 'Solo tactical retreat rejected'};return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory $OutputRoot|Out-Null
$runtime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1 -RuntimeRoot (Join-Path $OutputRoot 'runtime')
$server=$null;$bot=$null;$trace=Join-Path $OutputRoot 'bot.jsonl';$report=@{accepted=$false;reason='not_run';seed=$Seed}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_damage 1 +set g_test_seed $Seed +map base1"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $config=Join-Path $OutputRoot 'bot-config.json'
    $placement=if($Cover){'240,-416,24.125'}else{'32,-224,24'}
    $enemyClass=if($Cover){'monster_infantry'}else{'monster_parasite'}
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='SoloRetreatBot';game_dir=(Join-Path $runtime 'baseq2')};models=@{system1=$System1};run=@{duration='15s';frame_paced=$true;mode='campaign';next_map='base2'};test=@{teleport_map='base1';teleport=$placement;spawn_map='base1';spawn_soldier='200,-224,24';spawn_class=$enemyClass;setup_hold_frames=$(if($Cover){10}else{0});initial_health=$(if($Cover){25}else{0})};output=@{trace_jsonl=$trace}}|ConvertTo-Json -Depth 6|Set-Content $config
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $null=$bot.WaitForExit(25000);if(!$bot.HasExited){throw 'Bot timeout'};if($bot.ExitCode){throw 'Bot failed'}
    $rows=@(Get-Content $trace|ForEach-Object {$_|ConvertFrom-Json})
    $selected=@($rows|Where-Object {$_.tactic.source -eq 'live' -and $_.tactic.action -eq 'retreat'})
    $steps=@($selected|Where-Object {$_.arbitration.move_source -eq 'combat_retreat' -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0) -and ($_.sent_command.Buttons -band 1) -and $_.arbitration.aim_source -eq 'enemy'})
    $byFrame=@{};foreach($r in $rows){$byFrame[[int]$r.frame]=$r}
    $away=0;foreach($r in $steps){$next=$byFrame[([int]$r.frame+1)];$enemy=@($r.enemies|Where-Object id -eq $r.tactic.target);if(!$next -or $enemy.Count -ne 1){continue};$before=0.;$after=0.;foreach($i in 0..1){$before+=($r.self[$i]-$enemy[0].origin[$i])*($r.self[$i]-$enemy[0].origin[$i]);$after+=($next.self[$i]-$enemy[0].origin[$i])*($next.self[$i]-$enemy[0].origin[$i])};if([math]::Sqrt($after)-[math]::Sqrt($before) -gt 2){$away++}}
    $report.live_retreat_decisions=@($selected.tactic|Sort-Object map,frame -Unique).Count;$report.retreat_firing_frames=$steps.Count;$report.observed_away_steps=$away
    if(!$Cover -and (!$selected.Count -or $steps.Count -lt 2 -or $away -lt 2)){throw 'Model-selected firing retreat not observed'}
    if($Cover){
        $chosen=@($rows|Where-Object {$_.tactic.source -eq 'live' -and $_.tactic.action -eq 'cover'})
        $hidden=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_wait' -and @($_.enemies|Where-Object clear_shot).Count -eq 0})
        $fired=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_fire' -and ($_.sent_command.Buttons -band 1) -and $_.arbitration.aim_source -eq 'enemy'})
        $returned=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_return' -and $_.frame -gt $fired[0].frame -and @($_.enemies|Where-Object clear_shot).Count -eq 0})
        $report.model_cover_decisions=@($chosen.tactic|Sort-Object map,frame -Unique).Count;$report.cover_wait_frames=$hidden.Count;$report.cover_fire_frames=$fired.Count;$report.cover_return_frames=$returned.Count
        if(!$chosen.Count -or !$hidden.Count -or !$fired.Count -or !$returned.Count){throw 'Model-selected hide/fire/return cycle absent'}
        $report.scope='Prepared single infantry cover cycle; no group or general campaign acceptance'
    }
    if($rows|Where-Object {$_.map -ne 'base1' -or $_.teammate -or $_.health -le 0}){throw 'Unexpected map/teammate/death'}
    . "$PSScriptRoot/read_damage_events.ps1"
    $events=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log'));$actor=$rows[0].self_entity
    $report.damage_summary=Measure-BotDamage $events $actor
    $report.final_health=$rows[-1].health;$report.trace=$trace
    if($Cover){
        # Do not count earlier normal attack damage as proof of the cover shot.
        # The bounded 300-unit encounter allows at most three bolt flight frames.
        $coverDamage=@($events|Where-Object {
            $event=$_
            $event.attacker -eq $actor -and $event.target_class -eq $enemyClass -and $event.live_health_damage -gt 0 -and
            @($fired|Where-Object {$_.spawncount -eq $event.spawncount -and $event.frame -gt $_.frame -and $event.frame -le $_.frame+3}).Count -gt 0
        })
        $report.cover_window_health_damage=[int](($coverDamage|Measure-Object live_health_damage -Sum).Sum)
        $report.cover_window_damage_frames=@($coverDamage.frame)
    }
    $report.kills=@($events|Where-Object {$_.attacker -eq $actor -and $_.target_class -eq $enemyClass -and $_.killed}).Count
    if($report.kills -ne 1 -and !$Cover){throw 'Native target kill absent'}
    if(@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"}).Count -ne 1){throw 'Seed acknowledgement absent'}
    $commands=Get-Content (Join-Path $OutputRoot 'bot.err')
    if(@($commands|Select-String 'client command: teleport ').Count -ne 1 -or @($commands|Select-String 'client command: spawnentity ').Count -ne 1 -or ($commands -match 'client command: (god|kill|map|gamemap) ') -or @($commands|Select-String 'client command: give '|Where-Object { !$Cover -or $_.Line -notlike '*client command: give health 25' }).Count){throw 'Unexpected setup/gameplay command'}
    if($Cover -and $report.cover_window_health_damage -le 0){throw 'Native monster damage in cover firing window absent'};if(!$Cover){$report.scope='Prepared single vulnerable native parasite, ordinary solo campaign commands; model-selected retreat plus shooting and observed movement, not general campaign acceptance'};$report.accepted=$true;$report.reason='accepted'
}catch{$report.reason=$_.Exception.Message}
finally{foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}};$report|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')}
if(!$report.accepted){throw $report.reason}
