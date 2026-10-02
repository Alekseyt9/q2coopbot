[CmdletBinding()]
param([switch]$Worker,[switch]$Risk,[ValidateSet('weapon','ammo','covered','exposed','route')][string]$Case='weapon',[ValidateRange(0,2147483646)][int]$Seed=101,[int]$Port=31580,[string]$OutputRoot='',[string]$Client='',[string]$SourceRuntime='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$scenePath=Join-Path $PSScriptRoot $(if($Risk){'scenarios/base1-resource-risk-return.json'}else{'scenarios/base1-resource-memory-return.json'})
$scene=Get-Content $scenePath -Raw|ConvertFrom-Json
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $sceneHash=(Get-FileHash $scenePath).Hash
    $SourceRuntime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1
    $prefix=if($Risk){'resource-risk-return-'}else{'resource-memory-return-'}
    $OutputRoot=Join-Path $repo ('workspace/artifacts/'+$prefix+(Get-Date -Format yyyyMMdd-HHmmss-fff))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo
    try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $count=$scene.cases.Count*2
    $caseNames=@($scene.cases.name)
    $results=@(0..($count-1)|ForEach-Object -Parallel {
        $caseName=($using:caseNames)[[int][math]::Floor($_/2)]
        $out=Join-Path $using:OutputRoot "run-$_"
        $workerArgs=@('-NoProfile','-File',$using:script,'-Worker','-Case',$caseName,'-Port',($using:Port+$_),'-Seed',($using:Seed+($_%2)),'-OutputRoot',$out,'-Client',$using:Client,'-SourceRuntime',$using:SourceRuntime)
        if($using:Risk){$workerArgs+='-Risk'}
        $quotedArgs=@($workerArgs|ForEach-Object {'"'+$_+'"'})
        $child=Start-Process $using:hostExe -ArgumentList $quotedArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit()
        $reportPath=Join-Path $out 'report.json'
        if(Test-Path $reportPath){$r=Get-Content $reportPath -Raw|ConvertFrom-Json;if($child.ExitCode){$r.accepted=$false};$r}else{@{accepted=$false;reason='Worker did not produce report';case=$caseName}}
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -and $sceneHash -eq (Get-FileHash $scenePath).Hash
    $accepted=$valid -and $results.Count -eq $count -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;scene_sha256=$sceneHash;timescale=2;parallelism=2;seeds=@($Seed,($Seed+1));scope=$scene.scope;results=$results}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
    Write-Output "Resource memory return: $OutputRoot"
    if(!$accepted){throw 'Resource memory return rejected'}
    return
}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh trial output required'}
$spec=$scene.cases|Where-Object name -EQ $Case
if(!$spec){throw 'Case not present in selected scene'}
$runtime=Join-Path $OutputRoot 'runtime';$trace=Join-Path $OutputRoot 'bot.jsonl'
New-Item -ItemType Directory (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
Get-ChildItem $SourceRuntime -File|Where-Object Extension -In '.exe','.dll'|Copy-Item -Destination $runtime
Copy-Item (Join-Path $SourceRuntime 'baseq2/game.dll') (Join-Path $runtime 'baseq2/game.dll')
foreach($asset in Get-ChildItem (Join-Path $SourceRuntime 'baseq2') -File -Recurse|Where-Object Extension -In '.pak','.aas','.bsp','.ent'){
    $dest=Join-Path (Join-Path $runtime 'baseq2') ([IO.Path]::GetRelativePath((Join-Path $SourceRuntime 'baseq2'),$asset.FullName))
    New-Item -ItemType Directory (Split-Path $dest -Parent) -Force|Out-Null
    # Entity fixtures are private copies; never modify a hardlinked source .ent.
    if($asset.Extension -eq '.ent'){Copy-Item $asset.FullName $dest;continue}
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
$entPath=Join-Path $runtime 'baseq2/maps/base1.ent'
$blocks=@([regex]::Matches((Get-Content $entPath -Raw),'(?s)\{[^{}]*\}')|ForEach-Object Value)
$blocks=@($blocks|Where-Object {$_ -notmatch '"classname"\s+"(?:monster_|weapon_|ammo_|item_)'})
$blocks+="{`n"+'"classname" "'+$spec.class+"`"`n"+'"origin" "'+$scene.item_origin+"`"`n}"
if($spec.monster_origin){$blocks+="{`n"+'"classname" "monster_soldier"'+"`n"+'"origin" "'+$spec.monster_origin+"`"`n}"}
[IO.File]::WriteAllText($entPath,($blocks -join "`n"),[Text.Encoding]::ASCII)
$report=@{accepted=$false;reason='not_run';case=$Case;seed=$Seed;port=$Port;timescale=2;fixture_sha256=(Get-FileHash $entPath).Hash};$server=$null;$bot=$null
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_seed $Seed +map base1"
    if($Risk){$args='+set g_test_combat_barrier 1 '+$args}
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $test=@{teleport_map=$scene.map;teleport=$scene.origin;walk_target=$scene.walk_target;walk_route=$true;walk_then_plan=$true;walk_after_frames=$scene.walk_after_frames;walk_frames=$scene.walk_frames}
    if($spec.initial_fixture){$test.weapon_switch_fixture=$spec.initial_fixture}
    $config=Join-Path $OutputRoot 'bot-config.json'
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='MemoryBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='30s';frame_paced=$true;mode='campaign';next_map=$scene.next_map};output=@{trace_jsonl=$trace};test=$test}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $deadline=(Get-Date).AddSeconds(26);$last=$null
    do{
        $recent=@()
        if(Test-Path $trace){$recent=@(foreach($line in Get-Content $trace -Tail 12){try{$line|ConvertFrom-Json}catch{}});if($recent.Count){$last=$recent[-1]}}
        if($last.pickup.state -eq 'confirmed' -and $last.pickup.class -eq $spec.class){break}
        if($Risk -and $spec.expected -ne 'pickup' -and ($recent|Where-Object {$_.resource_risk.state -eq 'deferred' -and $_.resource_risk.reason -eq $spec.expected})){break}
        if($bot.HasExited -or $server.HasExited){throw 'Trial process exited'}
        Start-Sleep -Milliseconds 100
    }while((Get-Date) -lt $deadline)
    $rows=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}})
    $observed=$rows|Where-Object {($_.pickups|Where-Object class -EQ $spec.class)}|Select-Object -First 1
    $unknown=$rows|Where-Object {($_.resource_memory|Where-Object {$_.item.class -eq $spec.class -and $_.state -eq 'unknown'}) -and !($_.pickups|Where-Object class -EQ $spec.class)}|Select-Object -First 1
    $selected=$rows|Where-Object {$_.pickup.class -eq $spec.class -and $_.pickup.from_memory -and $_.pickup.state -eq 'approach'}|Select-Object -First 1
    $confirmed=$rows|Where-Object {$_.pickup.class -eq $spec.class -and $_.pickup.from_memory -and $_.pickup.state -eq 'confirmed'}|Select-Object -First 1
    $walk=@($rows|Where-Object {$_.arbitration.move_source -eq 'test_walk' -and ($_.sent_command.forward -ne 0 -or $_.sent_command.side -ne 0)})
    if(!$observed -or !$unknown -or !$walk.Count -or $unknown.frame -le $observed.frame){throw 'Seen, PVS loss or departure proof absent'}
    $origin=@($scene.origin.Split(',')|ForEach-Object {[double]::Parse($_,[cultureinfo]::InvariantCulture)})
    $teleportFrame=($rows|Where-Object {$_.self[0] -eq $origin[0] -and $_.self[1] -eq $origin[1]}|Select-Object -First 1).frame
    $scripted=@($rows|Where-Object {$_.arbitration.move_source -eq 'test_walk'})
    $releaseFrame=$scripted[-1].frame+1
    if(!$teleportFrame -or $releaseFrame -gt $teleportFrame+$scene.walk_after_frames+$scene.walk_frames+1){throw 'Unbounded walking prelude'}
    if($rows|Where-Object {$_.frame -ge $releaseFrame -and $_.arbitration.move_source -eq 'test_walk'}){throw 'Scripted movement after planner release'}
    $danger=$null
    if($Risk -and $spec.expected -ne 'pickup'){
        $danger=$rows|Where-Object {$_.frame -ge $releaseFrame -and $_.resource_risk.state -eq 'deferred' -and $_.resource_risk.reason -eq $spec.expected}|Select-Object -First 1
        if(!$danger -or $danger.pickup.state -eq 'approach' -or $danger.pickup.state -eq 'confirmed' -or $danger.goal -ne 'reach_level_exit' -or !($danger.enemies|Where-Object id -EQ $danger.resource_risk.enemy)){throw 'Deferred threat and preserved campaign objective proof absent'}
        if($selected -and $selected.frame -le $danger.frame){throw 'Unsafe memory return selected before defer'}
        $memory=$danger.resource_memory|Where-Object {$_.item.class -eq $spec.class}
        if($memory.attempted -or $memory.state -ne 'unknown'){throw 'Unsafe route consumed the resource memory'}
    }else{
        if(!$selected -or !$confirmed -or $selected.frame -lt $releaseFrame -or $selected.frame -le $walk[-1].frame -or $confirmed.pickup.after -le $confirmed.pickup.before){throw 'Memory selection or inventory delta proof absent'}
        if($Case -eq 'covered'){
            $safe=$rows|Where-Object {$_.pickup.from_memory -and $_.resource_risk.state -eq 'allowed' -and $_.resource_risk.reason -eq 'no_route_exposure' -and $_.enemies}|Select-Object -First 1
            if(!$safe){throw 'Observed covered monster and allowed detour proof absent'}
            $report.covered_route=$safe.resource_risk
        }
    }
    if($rows|Where-Object {$_.teammate -or $_.health -le 0 -or $_.map -ne $scene.map}){throw 'Unexpected companion, death or map transition'}
    $log=Get-Content (Join-Path $OutputRoot 'bot.err') -Raw
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne 1 -or $log -match 'client command: (map|gamemap|kill) '){throw 'Forced gameplay command'}
    if($Case -ne 'ammo' -and $log -match 'client command: give '){throw 'Weapon trial used give'}
    if($Risk -and (Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -like 'g_test_combat_start *'})){throw 'Diagnostic unexpectedly released native monster AI'}
    $seedAck=@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"})
    if($seedAck.Count -ne 1){throw 'Native seed acknowledgement absent'}
    $report.accepted=$true;$report.reason='accepted';$report.seed_verified=$true;$report.observed_frame=$observed.frame;$report.unseen_frame=$unknown.frame;$report.selection_frame=$selected.frame;$report.release_frame=$releaseFrame;$report.confirmation=$confirmed.pickup;$report.trace=$trace;$report.scripted_walk_frames=$walk.Count
    if($danger){$report.deferred_frame=$danger.frame;$report.risk=$danger.resource_risk;$report.goal=$danger.goal}
}catch{$report.accepted=$false;$report.reason=$_.Exception.Message}finally{
    foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue}}
    foreach($p in @($bot,$server)){if($p){$null=$p.WaitForExit(5000)}}
    $report|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
}
if(!$report.accepted){throw $report.reason}
