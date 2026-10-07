[CmdletBinding()]
param(
    [string]$Model='',
    [Parameter(Mandatory)][string]$OutputRoot,
    [int]$Seed=44900,[int]$Port=34100,
    [ValidateRange(1,8)][int]$Workers=4,
    [ValidateRange(100,10000)][int]$GameFrames=1800,
    [ValidateRange(0,3)][int]$Skill=1,
    [ValidateSet('base1','base2')][string[]]$Maps=@('base1','base2'),
    [ValidateRange(1,128)][int]$EpisodesPerMap=4,
    [ValidateSet('rules','learned')][string[]]$Modes=@('rules','learned')
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
. "$PSScriptRoot/harness_manifest.ps1"
$fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$runtimes=@{}
foreach($map in $Maps){
    $runtimes[$map]=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map $map -KeepMonsters
}
$sourceModel='';$modelHash='';$evalModel=''
if('learned' -in $Modes -and !$Model){throw 'Learned campaign needs a model'}
if($Model){
    $sourceModel=(Resolve-Path -LiteralPath $Model).Path
    $modelHash=(Get-FileHash -LiteralPath $sourceModel).Hash.ToLowerInvariant()
    $weights=Get-Content -LiteralPath $sourceModel -Raw|ConvertFrom-Json -AsHashtable
    $weights.deterministic=$true
    $evalModel=Join-Path $OutputRoot 'deterministic-weights.json'
    $weights|ConvertTo-Json -Depth 100 -Compress|Set-Content -LiteralPath $evalModel -Encoding utf8
}
$client=Join-Path $OutputRoot 'q2coopbot.exe'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Build failed'}}finally{Pop-Location}
$tasks=@()
foreach($map in $Maps){
    foreach($mode in $Modes){
        foreach($index in 0..($EpisodesPerMap-1)){
            $taskPort=$Port+$tasks.Count
            if(Get-NetUDPEndpoint -LocalPort $taskPort -ErrorAction SilentlyContinue){throw "Occupied port $taskPort"}
            $runSeed=$Seed+$index+$(if($map -eq 'base2'){100}else{0})
            $dir=Join-Path $OutputRoot "$map-$mode-$runSeed"
            $runtime=Join-Path $dir 'runtime'
            New-Item -ItemType Directory -Path (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
            $source=$runtimes[$map]
            Get-ChildItem -LiteralPath $source -File|Where-Object Extension -In '.exe','.dll'|Copy-Item -Destination $runtime
            Copy-Item -LiteralPath (Join-Path $source 'baseq2/game.dll') -Destination (Join-Path $runtime 'baseq2/game.dll')
            foreach($asset in Get-ChildItem (Join-Path $source 'baseq2') -File -Recurse|Where-Object Extension -In '.pak','.aas','.bsp','.ent'){
                $dest=Join-Path (Join-Path $runtime 'baseq2') ([IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName))
                New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force|Out-Null
                New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null
            }
            $tasks+=@{map=$map;mode=$mode;seed=$runSeed;port=$taskPort;dir=$dir;runtime=$runtime}
        }
    }
}
@{source_model=$sourceModel;source_model_sha256=$modelHash;evaluation_model_sha256=$(if($evalModel){(Get-FileHash $evalModel).Hash.ToLowerInvariant()}else{''});deterministic=$true;source_fingerprint=$fingerprint;workers=$Workers;timescale=2;skill=$Skill;game_frames=$GameFrames;tasks=$tasks}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'manifest.json') -Encoding utf8
$worker={
    param($task,$client,$evalModel,$frames,$skill,$scriptRoot)
    $ErrorActionPreference='Stop'
    . "$scriptRoot/read_damage_events.ps1"
    $server=$null;$bot=$null
    $trace=Join-Path $task.dir 'bot.jsonl'
    $stop=Join-Path $task.dir 'stop.txt'
    $watch=[diagnostics.stopwatch]::StartNew()
    $report=@{map=$task.map;mode=$task.mode;seed=$task.seed;infrastructure_ok=$false;completed=$false;reason='not_started'}
    try{
        $next=if($task.map -eq 'base1'){'base2'}else{'base3'}
        $serverArgs="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 0 +set maxclients 4 +set port $($task.port) +set timescale 2 +set skill $skill +set g_test_seed $($task.seed) +set g_test_damage 1 +set sv_test_unlimited_loopback 1 +map $($task.map)"
        $server=Start-Process (Join-Path $task.runtime 'q2ded.exe') -ArgumentList $serverArgs -WorkingDirectory $task.runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $task.dir 'server.log') -RedirectStandardError (Join-Path $task.dir 'server.err')
        $deadline=(Get-Date).AddSeconds(15)
        do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $task.port -ErrorAction SilentlyContinue))
        if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Non-loopback listener'}
        $combat=@{mode=$task.mode}
        if($task.mode -eq 'learned'){$combat.provider_file=$evalModel}
        $config=Join-Path $task.dir 'config.json'
        @{server=@{host='127.0.0.1';port=$task.port};client=@{name='CampaignBot';game_dir=(Join-Path $task.runtime 'baseq2')};combat=$combat;run=@{mode='campaign';campaign_route=@($task.map,$next);frame_paced=$true;game_frames=$frames;duration='240s'};output=@{trace_jsonl=$trace;combat_capture=$true;record_demo=$false;stop_file=$stop};test=@{campaign_combat_evaluation=$true}}|ConvertTo-Json -Depth 8|Set-Content $config -Encoding utf8
        $bot=Start-Process $client -ArgumentList @('--config',('"'+$config+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $task.dir 'bot.log') -RedirectStandardError (Join-Path $task.dir 'bot.err')
        $deadline=(Get-Date).AddSeconds(260)
        while(!$bot.HasExited){
            if((Get-Date) -gt $deadline){throw 'Client wall timeout'}
            if(Test-Path $trace){
                $tail=Get-Content -LiteralPath $trace -Tail 2
                foreach($line in $tail){
                    try{$row=$line|ConvertFrom-Json}catch{continue}
                    if($row.map -and $row.map -ne $task.map){
                        $report.exit_map=$row.map
                        $report.completed=$row.map -eq $next
                        $report.reason=if($report.completed){'native_map_exit'}else{'native_wrong_exit'}
                        Set-Content -LiteralPath $stop -Value $report.reason
                        break
                    }
                }
            }
            Start-Sleep -Milliseconds 500
        }
        $report.client_exit_code=$bot.ExitCode
        if($bot.ExitCode){throw "Client failed: $($bot.ExitCode)"}
        $report.infrastructure_ok=$true
        if($report.reason -eq 'not_started'){$report.reason='frame_budget'}
    }catch{$report.reason=$_.Exception.Message}
    finally{
        foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue};if($p){$null=$p.WaitForExit(5000)}}
        $report.wall_seconds=$watch.Elapsed.TotalSeconds
        try{
            $rows=@(Get-Content -LiteralPath $trace|ForEach-Object {try{$r=$_|ConvertFrom-Json;if($r.map -eq $task.map){$r}}catch{}})
            if(!$rows.Count){throw 'No map trace'}
            $unique=@($rows|Group-Object spawncount,frame|ForEach-Object {$_.Group[0]})
            $deaths=0;$prior=$unique[0]
            foreach($r in $unique){if($prior.health -gt 0 -and $r.health -le 0){$deaths++};$prior=$r}
            $events=@(Read-DamageEvents (Join-Path $task.dir 'server.log')|Where-Object map -eq $task.map)
            $actor=$rows[0].self_entity
            $damage=Measure-BotDamage $events $actor
            $report.frames=$unique.Count;$report.first_frame=$unique[0].frame;$report.last_frame=$unique[-1].frame
            $report.game_seconds=($unique[-1].frame-$unique[0].frame)/10
            $report.deaths=$deaths;$report.initial_health=$unique[0].health;$report.final_health=$unique[-1].health
            $report.final_position=$unique[-1].self;$report.last_goal=$unique[-1].goal;$report.last_campaign=$unique[-1].campaign
            $report.monsters_killed=@($events|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*' -and $_.killed}).Count
            $report.monster_health_damage=[int](($events|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*'}|Measure-Object live_health_damage -Sum).Sum)
            $report.received_health_damage=$damage.received_health_damage
            $report.damage_by_weapon=$damage.by_mod
            $report.weapons=@($rows.weapon|Sort-Object -Unique)
            $report.pickups=@($rows|Where-Object {$_.pickup.confirmed}).Count
            $selections=@($unique|ForEach-Object {$_.combat_policy.selection}|Where-Object {$_})
            $provider=@($selections|Where-Object owner -eq 'provider')
            $report.provider_frames=$provider.Count;$report.provider_percent=[math]::Round(100*$provider.Count/[math]::Max(1,$selections.Count),2)
            $report.fallbacks=@($selections|Where-Object fallback|Group-Object fallback|ForEach-Object {@{reason=$_.Name;frames=$_.Count}})
            $report.inference_max_us=($selections|Measure-Object elapsed_us -Maximum).Maximum
            $report.inference_mean_us=($provider|Measure-Object elapsed_us -Average).Average
        }catch{$report.infrastructure_ok=$false;$report.stats_error=$_.Exception.Message}
        $report|ConvertTo-Json -Depth 12|Set-Content (Join-Path $task.dir 'report.json') -Encoding utf8
    }
    [pscustomobject]$report
}
$workerText=$worker.ToString()
$results=@($tasks|ForEach-Object -Parallel {
    $runWorker=[scriptblock]::Create($using:workerText)
    & $runWorker $_ $using:client $using:evalModel $using:GameFrames $using:Skill $using:PSScriptRoot
} -ThrottleLimit $Workers)
$summary=@{source_unchanged=($fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)));source_model_sha256=$modelHash;timescale=2;skill=$Skill;workers=$Workers;results=$results}
$summary|ConvertTo-Json -Depth 14|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
Write-Output $OutputRoot
if(@($results|Where-Object {!$_.infrastructure_ok}).Count){throw 'Comparison includes infrastructure failures; inspect per-run reports'}
