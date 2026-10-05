[CmdletBinding()]
param(
    [ValidateRange(2,4)][int]$Workers=4,
    [ValidateRange(1,20)][int]$EpisodesPerWorker=2,
    [ValidateSet(1,2)][int]$Timescale=2,
    [ValidateRange(20,500)][int]$GameFrames=300,
    [int]$Seed=9300,
    [ValidateRange(1024,65530)][int]$Port=32940,
    [ValidateSet('stocked','blaster','hyper','rail','scarce')][string]$Loadout='stocked',
    [switch]$Mixed,
    [ValidateSet('rules','learned-shadow','learned')][string]$CombatMode='rules',
    [string]$ProviderFile='',
    [string]$OutputRoot=''
)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/learned-combat-baseline-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output directory required'}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
if($CombatMode -ne 'rules' -and $Loadout -ne 'blaster'){throw 'Direct/shadow pilot requires -Loadout blaster'}
if($CombatMode -ne 'rules' -and !$ProviderFile){$ProviderFile=Join-Path $PSScriptRoot 'scenarios/combat-control-probe.json'}
if($ProviderFile){$ProviderFile=(Resolve-Path -LiteralPath $ProviderFile).Path}
$sources=Get-HarnessSourceRecords $repo
$fingerprint=Get-HarnessFingerprint $sources
$client=Join-Path $OutputRoot 'q2coopbot.exe'
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
$env:GOTOOLCHAIN='auto'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
$exporter=Join-Path $OutputRoot 'q2combat-export.exe'
Push-Location $repo;try{go build -o $exporter ./cmd/q2combat-export;if($LASTEXITCODE){throw 'Exporter build failed'}}finally{Pop-Location}
$probeHash=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{''})
$hostExe=(Get-Process -Id $PID).Path
$runner=Join-Path $PSScriptRoot 'run_solo_tactical_retreat.ps1'
$clock=[Diagnostics.Stopwatch]::StartNew()
$results=@(0..($Workers-1) | ForEach-Object -Parallel {
    $worker=$_
    for($episode=0;$episode -lt $using:EpisodesPerWorker;$episode++){
        $index=$worker*$using:EpisodesPerWorker+$episode
        $out=Join-Path $using:OutputRoot "worker-$worker-episode-$episode"
        $arguments=@('-NoProfile','-File',$using:runner,'-Worker','-Rules','-CombatCapture','-ParasiteWeapon',
            '-ParasiteLoadout',$using:Loadout,'-Seed',($using:Seed+$index),'-Port',($using:Port+$worker),
            '-Timescale',$using:Timescale,'-OutputRoot',$out,'-Client',$using:client)
        $arguments+=@('-CombatMode',$using:CombatMode);if($using:ProviderFile){$arguments+=@('-ProviderFile',$using:ProviderFile)}
        $arguments+=@('-GameFrames',$using:GameFrames)
        if($using:Mixed){$arguments+=@('-ParasiteMixed','-ParasiteMixedClass','monster_gunner')}
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $process=Start-Process $using:hostExe -ArgumentList @($arguments | ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.stdout.log" -RedirectStandardError "$out.stderr.log"
        $process.WaitForExit()
        $timer.Stop()
        try{
            $report=Get-Content -LiteralPath (Join-Path $out 'report.json') -Raw | ConvertFrom-Json
            $rows=@(Get-Content -LiteralPath (Join-Path $out 'bot.jsonl') | ForEach-Object {$_ | ConvertFrom-Json})
            if(!$rows.Count){throw 'Empty command trace'}
            $captures=@($rows | Where-Object combat_policy)
            $mismatches=@($rows | Where-Object {
                !$_.combat_policy -or ($_.combat_policy.provider -ne 'rules' -and $_.combat_policy.selection.owner -ne 'provider') -or
                $_.combat_policy.observation.version -ne 'combat_observation_v2' -or
                $_.combat_policy.applied.version -ne 'combat_action_v1' -or
                $_.combat_policy.observation.identity.frame -ne $_.observation_frame -or
                $_.combat_policy.observation.identity.connection -ne $_.connection -or
                $_.combat_policy.observation.identity.spawncount -ne $_.spawncount -or
                $_.combat_policy.observation.identity.actor -ne $_.self_entity -or
                $_.combat_policy.observation.identity.map -ne $_.map -or
                ($_.combat_policy.applied_command | ConvertTo-Json -Compress) -ne ($_.sent_command | ConvertTo-Json -Compress)
            }).Count
            $ns=@($captures.combat_policy.command_at_unix_ns)
            $activeSeconds=if($ns.Count -gt 1){([double]$ns[-1]-[double]$ns[0])/1e9}else{0}
            $summary=Select-String -LiteralPath (Join-Path $out 'bot.err') -Pattern 'finished connected=.*' | Select-Object -Last 1
            if(!$summary){throw 'Client completion summary missing'}
            $summaryText=$summary.Line
            $field=@{}
            foreach($match in [regex]::Matches($summaryText,'(game_fps|wall_s|frame_gaps|decode_errors|game_frames|moves)=([0-9.]+)')){$field[$match.Groups[1].Value]=[double]::Parse($match.Groups[2].Value,[Globalization.CultureInfo]::InvariantCulture)}
            $seedAck=@(Get-Content -LiteralPath (Join-Path $out 'server.log') | Where-Object {$_ -eq "g_test_seed ready version=1 seed=$($using:Seed+$index)"}).Count -eq 1
            $dataset=Join-Path $out 'dataset'
            $config=Get-Content -LiteralPath (Join-Path $out 'bot-config.json') -Raw|ConvertFrom-Json
            $resetExpectation=Join-Path $out 'reset-expectation.json'
            @{version='observed_fixture_reset_v1';map=$config.test.teleport_map
                position=@($config.test.teleport.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
                health=$config.test.initial_health;armor=0;weapon='Shotgun';ammo=20;enemy_class=$config.test.spawn_class
                enemy_position=@($config.test.spawn_soldier.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
            }|ConvertTo-Json|Set-Content -LiteralPath $resetExpectation -Encoding utf8NoBOM
            & $using:exporter --trace (Join-Path $out 'bot.jsonl') --out $dataset --worker "worker-$worker" --episode "seed-$($using:Seed+$index)" --end-reason game_frame_limit --server-log (Join-Path $out 'server.log') --client-name SoloRetreatBot --require-execution --reset-expectation $resetExpectation
            if($LASTEXITCODE){throw 'Transition export failed'}
            $datasetReport=Get-Content -LiteralPath (Join-Path $dataset 'report.json') -Raw|ConvertFrom-Json
            $controlled=@($captures|Where-Object {$_.combat_policy.selection.owner -eq 'provider'}).Count
            $shadow=@($captures|Where-Object {$_.combat_policy.selection.mode -eq 'learned-shadow' -and $_.combat_policy.selection.candidate_command}).Count
            $changes=@(foreach($capture in $captures){foreach($change in @($capture.combat_policy.selection.interventions)){if($change -and $change.reason){$change}}})
            $latencies=@($captures|Where-Object {$_.combat_policy.selection.candidate_command}|ForEach-Object {[long]$_.combat_policy.selection.elapsed_us}|Sort-Object)
            $p95=$(if($latencies.Count){$latencies[[math]::Ceiling(.95*$latencies.Count)-1]}else{$null})
            $p99=$(if($latencies.Count){$latencies[[math]::Ceiling(.99*$latencies.Count)-1]}else{$null})
            $dispatchValid=($using:CombatMode -eq 'rules') -or (($using:CombatMode -eq 'learned') -and ($controlled -gt 0)) -or (($using:CombatMode -eq 'learned-shadow') -and ($shadow -gt 0) -and ($controlled -eq 0))
            $life=Get-Content -LiteralPath (Join-Path $out 'combat-first-life.json') -Raw | ConvertFrom-Json
            $files=@('q2ded.exe','baseq2/game.dll','baseq2/maps/base1.ent','baseq2/maps/base1.aas') | ForEach-Object {
                $path=Join-Path (Join-Path $out 'runtime') $_
                if(Test-Path -LiteralPath $path){@{path=$_;sha256=(Get-FileHash -LiteralPath $path).Hash}}
            }
            [pscustomobject]@{
                worker=$worker;episode=$episode;seed=($using:Seed+$index);port=($using:Port+$worker);seed_confirmed=$seedAck
                harness_accepted=[bool]$report.accepted;harness_reason=$report.reason;worker_exit_code=$process.ExitCode
                capture_valid=($captures.Count -eq $rows.Count -and $mismatches -eq 0 -and $field.decode_errors -eq 0 -and $field.game_frames -eq $using:GameFrames -and $seedAck -and $dispatchValid -and $datasetReport.command_proof.accepted -and $datasetReport.observed_reset_confirmed)
                dispatch_valid=$dispatchValid;dataset=$datasetReport;dataset_bytes=(Get-Item -LiteralPath (Join-Path $dataset 'steps.jsonl')).Length
                capture_mismatches=$mismatches
                provider_controlled_frames=@($captures|Where-Object {$_.combat_policy.selection.owner -eq 'provider'}).Count
                shadow_candidate_frames=@($captures|Where-Object {$_.combat_policy.selection.mode -eq 'learned-shadow' -and $_.combat_policy.selection.candidate_command}).Count
                guard_interventions=$changes.Count
                selection_p95_us=$p95;selection_p99_us=$p99;selection_max_us=$(if($latencies.Count){$latencies[-1]}else{$null})
                commands=$rows.Count;active_seconds=$activeSeconds;client_wall_seconds=$field.wall_s;game_fps=$field.game_fps
                decision_fps=$(if($activeSeconds -gt 0){($rows.Count-1)/$activeSeconds}else{0})
                frame_gaps=$field.frame_gaps;decode_errors=$field.decode_errors
                cold_episode_seconds=$timer.Elapsed.TotalSeconds
                reset_to_udp_ready_seconds=$report.reset_to_udp_ready_seconds
                setup_and_teardown_seconds=($timer.Elapsed.TotalSeconds-$field.wall_s)
                trace_bytes=(Get-Item -LiteralPath (Join-Path $out 'bot.jsonl')).Length
                log_bytes=[long]( (Get-ChildItem -LiteralPath $out -File | Where-Object {$_.Extension -in '.log','.err','.jsonl'} | Measure-Object Length -Sum).Sum )
                first_life=$life;runtime_files=$files;root=$out
            }
        }catch{
            [pscustomobject]@{worker=$worker;episode=$episode;seed=($using:Seed+$index);capture_valid=$false;error=$_.Exception.Message;worker_exit_code=$process.ExitCode;root=$out;cold_episode_seconds=$timer.Elapsed.TotalSeconds}
        }
    }
} -ThrottleLimit $Workers)
$clock.Stop()
$valid=($fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)))
if($ProviderFile){$valid=$valid -and $probeHash -eq (Get-FileHash -LiteralPath $ProviderFile).Hash}
$usable=@($results | Where-Object capture_valid)
$manifest=[ordered]@{
    version=2;stage='R1 dispatch and R2 transition pilot';provider=$CombatMode;model_weights=$null;probe_sha256=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{$null})
    observation_version='combat_observation_v2';action_version='combat_action_v1';reward_version=$null;server_outcome_version='server_damage_window_v1'
    timescale=$Timescale;game_frames=$GameFrames;workers=$Workers;episodes_per_worker=$EpisodesPerWorker;loadout=$Loadout;mixed=[bool]$Mixed
    reset='Cold native server restart per episode, verified first usable observed fixture fields. Inventory/RNG/AI/full-world equivalence remain unconfirmed.'
    source_fingerprint=$fingerprint;provenance_valid=$valid;sources=$sources
    client_sha256=(Get-FileHash -LiteralPath $client).Hash
    plan_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'docs/learned_system1_plan.md')).Hash
    bsp_assets=@(Get-HarnessFileRecords -Root (Join-Path (Split-Path $repo -Parent) 'assets/baseq2') -Paths @((Join-Path (Split-Path $repo -Parent) 'assets/baseq2/pak0.pak')))
    seeds=@($results.seed);physics='Stock native physics; only timescale and loopback rate differ. Live params remain in worker config, without RCON.'
    stop='Fixed game-frame limit per client; 60s wall watchdog; first-life diagnostics and per-life transitions remain separate.'
    dataset_scope='Separate observed steps, exact native command dispatch proof and damage effect windows; effects are not shot accuracy or delayed causal credit. No scalar reward or positive demonstration labels. Provider probe is not trained.'
}
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $OutputRoot 'manifest.json') -Encoding utf8
$report=[ordered]@{
    version=1;provenance_valid=$valid;capture_complete=($valid -and $usable.Count -eq $Workers*$EpisodesPerWorker)
    timescale=$Timescale;parallelism=$Workers;wall_seconds=$clock.Elapsed.TotalSeconds
    completed_episodes=$results.Count;usable_captures=$usable.Count
    aggregate_decisions_per_second=([double](($usable | Measure-Object commands -Sum).Sum)/$clock.Elapsed.TotalSeconds)
    episodes_per_second=($results.Count/$clock.Elapsed.TotalSeconds)
    gameplay_accepted=@($results | Where-Object harness_accepted).Count
    results=$results
}
$report | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath (Join-Path $OutputRoot 'report.json') -Encoding utf8
"Baseline: $OutputRoot"
if(!$report.capture_complete){throw 'Baseline capture/provenance incomplete; inspect preserved reports'}
