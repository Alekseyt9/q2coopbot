[CmdletBinding()]
param(
    [ValidateRange(2,4)][int]$Workers=4,
    [ValidateRange(1,20)][int]$EpisodesPerWorker=2,
    [ValidateSet(1,2)][int]$Timescale=2,
    [ValidateRange(20,500)][int]$GameFrames=300,
    [int]$Seed=9300,
    [ValidateSet(0,10,20,30,40,60,100)][int]$TrainingMonsterHealth=0,
    [ValidateSet(0,100)][int]$ReleaseGameFrame=0,
    [ValidateRange(1024,65530)][int]$Port=32940,
    [ValidateSet('stocked','blaster','shotgun','hyper','rail','scarce')][string]$Loadout='stocked',
    [switch]$Mixed,
    [switch]$HealthKit,
    [switch]$Synchronous,
    [switch]$KeepRuntimeAssets,
    [switch]$TeacherVertical,
    [switch]$Feedback,
    [ValidateSet('rules','learned-shadow','learned')][string]$CombatMode='rules',
    [string]$ProviderFile='',
    [string]$RewardConfig='',
    [string]$OutputRoot=''
)
$ErrorActionPreference='Stop'
if($Seed -lt 0 -or [long]$Seed+$Workers*$EpisodesPerWorker-1 -gt 2147483647){throw 'Each independent episode needs its own valid 31-bit game seed'}
if($Synchronous -and $Loadout -notin 'blaster','shotgun'){throw 'Synchronous learning requires Blaster or rules Shotgun exercise'}
if($Loadout -eq 'shotgun' -and (!$Synchronous -or $CombatMode -ne 'rules')){throw 'Shotgun exercise requires synchronous rules'}
if($TeacherVertical -and (!$Synchronous -or $CombatMode -ne 'rules' -or $Loadout -ne 'shotgun' -or $Feedback)){throw 'Vertical exercise requires synchronous fixed Shotgun rules without feedback'}
if($RewardConfig -and !$Synchronous){throw 'Reward export requires -Synchronous'}
if($TrainingMonsterHealth -and (!$Synchronous -or $Loadout -ne 'blaster' -or $Mixed -or $HealthKit -or $Feedback)){throw 'Curriculum requires isolated synchronous Blaster fixture'}
if($ReleaseGameFrame -and (!$Synchronous -or $Loadout -ne 'blaster' -or $HealthKit -or $GameFrames -lt 150)){throw 'Fixed release requires synchronous Blaster fixture without health kit and >=150 game frames'}
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/learned-combat-baseline-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output directory required'}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
if($CombatMode -ne 'rules' -and $Loadout -ne 'blaster'){throw 'Direct/shadow pilot requires -Loadout blaster'}
if($CombatMode -ne 'rules' -and !$ProviderFile){$ProviderFile=Join-Path $PSScriptRoot 'scenarios/combat-control-probe.json'}
if($ProviderFile){$ProviderFile=(Resolve-Path -LiteralPath $ProviderFile).Path}
$remoteProvider=$false
$providerKind=$null
if($ProviderFile){$providerKind=(Get-Content -LiteralPath $ProviderFile -Raw|ConvertFrom-Json).kind;$remoteProvider=($providerKind -eq 'combat_remote_v1')}
$ppoProvider=($providerKind -eq 'combat_ppo_v1')
$trainedProvider=$providerKind -in @('combat_bc_mlp_v1','combat_ppo_v1')
if($ppoProvider -and !$Synchronous){throw 'PPO pilot requires synchronous mode'}
if($remoteProvider -and !$Synchronous){throw 'Remote policy requires -Synchronous'}
if($Feedback -and (!$remoteProvider -or !$RewardConfig)){throw 'Feedback requires remote provider, synchronous mode and reward config'}
if($RewardConfig){$RewardConfig=(Resolve-Path -LiteralPath $RewardConfig).Path}
$rewardHash=$(if($RewardConfig){(Get-FileHash -LiteralPath $RewardConfig).Hash}else{''})
$sources=Get-HarnessSourceRecords $repo
$fingerprint=Get-HarnessFingerprint $sources
$nativeRepo=Join-Path (Split-Path $repo -Parent) 'yquake2'
$nativeSources=Get-HarnessNativeSourceRecords $nativeRepo
$nativeFingerprint=Get-HarnessFingerprint $nativeSources
$client=Join-Path $OutputRoot 'q2coopbot.exe'
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
$env:GOTOOLCHAIN='auto'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
$exporter=Join-Path $OutputRoot 'q2combat-export.exe'
Push-Location $repo;try{go build -o $exporter ./cmd/q2combat-export;if($LASTEXITCODE){throw 'Exporter build failed'}}finally{Pop-Location}
$relay=Join-Path $OutputRoot 'q2learning-relay.exe'
if($Feedback){Push-Location $repo;try{go build -o $relay ./cmd/q2learning-relay;if($LASTEXITCODE){throw 'Relay build failed'}}finally{Pop-Location}}
$probeHash=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{''})
$hostExe=(Get-Process -Id $PID).Path
$runner=Join-Path $PSScriptRoot 'run_solo_tactical_retreat.ps1'
$feedbackAudit=Join-Path $PSScriptRoot 'audit_learning_feedback.ps1'
$clock=[Diagnostics.Stopwatch]::StartNew()
$results=@(0..($Workers-1) | ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    $worker=$_
    for($episode=0;$episode -lt $using:EpisodesPerWorker;$episode++){
        $index=$worker*$using:EpisodesPerWorker+$episode
        $out=Join-Path $using:OutputRoot "worker-$worker-episode-$episode"
		$episodeProvider=$using:ProviderFile
		if($using:remoteProvider -or $using:ppoProvider){
			$template=Get-Content -LiteralPath $using:ProviderFile -Raw|ConvertFrom-Json
			if($using:remoteProvider){$template.episode="worker-$worker-seed-$($using:Seed+$index)";$template.seed=$using:Seed+$index}
			if($using:ppoProvider){$template.sampling_seed=$using:Seed+$index}
			$episodeProvider=Join-Path $using:OutputRoot "provider-worker-$worker-episode-$episode.json"
			$template|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $episodeProvider -Encoding utf8NoBOM
		}
		$episodeProviderHash=$(if($episodeProvider){(Get-FileHash -LiteralPath $episodeProvider).Hash}else{$null})
		$feedbackProcess=$null
		$feedbackRoot=Join-Path $using:OutputRoot "feedback-worker-$worker-episode-$episode"
		$feedbackStop="$feedbackRoot.stop"
		if($using:Feedback){
			$feedbackReset="$feedbackRoot.reset.json"
			@{version='observed_fixture_reset_v1';map='base1';seed=$using:Seed+$index;position=@(32,-224,24);health=100;armor=0;weapon='Blaster';ammo=0;enemy_class='monster_parasite';enemy_position=@(200,-224,24)}|ConvertTo-Json|Set-Content -LiteralPath $feedbackReset -Encoding utf8NoBOM
            $relayArguments=@('--trace',(Join-Path $out 'bot.jsonl'),'--server-log',(Join-Path $out 'server.log'),'--provider',$episodeProvider,'--reward-config',$using:RewardConfig,'--reset-expectation',$feedbackReset,'--out',$feedbackRoot,'--stop-file',$feedbackStop,'--worker',"worker-$worker",'--episode',"seed-$($using:Seed+$index)",'--game-frames',"$($using:GameFrames)")
			$feedbackProcess=Start-Process $using:relay -ArgumentList @($relayArguments|ForEach-Object{'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$feedbackRoot.out" -RedirectStandardError "$feedbackRoot.err"
		}
        $arguments=@('-NoProfile','-File',$using:runner,'-Worker','-Rules','-CombatCapture','-ParasiteWeapon',
            '-ParasiteLoadout',$using:Loadout,'-Seed',($using:Seed+$index),'-Port',($using:Port+$worker),
            '-Timescale',$using:Timescale,'-OutputRoot',$out,'-Client',$using:client)
        $arguments+=@('-CombatMode',$using:CombatMode);if($episodeProvider){$arguments+=@('-ProviderFile',$episodeProvider)}
        $arguments+=@('-GameFrames',$using:GameFrames)
        $arguments+=@('-TrainingMonsterHealth',$using:TrainingMonsterHealth)
        $arguments+=@('-ReleaseGameFrame',$using:ReleaseGameFrame)
        if($using:Synchronous){$arguments+='-Synchronous'}
        if($using:TeacherVertical){$arguments+='-TeacherVertical'}
        if($using:Mixed){$arguments+=@('-ParasiteMixed','-ParasiteMixedClass','monster_gunner')}
        if($using:HealthKit){$arguments+='-ParasiteHealthKit'}
        $timer=[Diagnostics.Stopwatch]::StartNew()
        try{
            $process=Start-Process $using:hostExe -ArgumentList @($arguments | ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$out.stdout.log" -RedirectStandardError "$out.stderr.log"
            $process.WaitForExit()
        }finally{
            if($feedbackProcess){
                Set-Content -LiteralPath $feedbackStop -Value 'worker_complete' -Encoding ascii
                if(!$feedbackProcess.WaitForExit(5000)){Stop-Process -Id $feedbackProcess.Id}
            }
        }
        $timer.Stop()
        try{
            $feedbackReport=$null
            if($using:Feedback){$feedbackReport=Get-Content -LiteralPath (Join-Path $feedbackRoot 'report.json') -Raw|ConvertFrom-Json;if(!$feedbackReport.accepted -or $feedbackProcess.ExitCode){throw 'Live feedback delivery failed'}}
            $report=Get-Content -LiteralPath (Join-Path $out 'report.json') -Raw | ConvertFrom-Json
            $rows=@(Get-Content -LiteralPath (Join-Path $out 'bot.jsonl') | ForEach-Object {$_ | ConvertFrom-Json})
            if(!$rows.Count){throw 'Empty command trace'}
            $captures=@($rows | Where-Object combat_policy)
            $mismatches=@($rows | Where-Object {
                !$_.combat_policy -or ($_.combat_policy.provider -ne 'rules' -and $_.combat_policy.selection.owner -ne 'provider') -or
                $_.combat_policy.observation.version -ne 'combat_observation_v3' -or
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
            @{version='observed_fixture_reset_v1';map=$config.test.teleport_map;seed=$(if($using:Synchronous){$using:Seed+$index}else{$null})
                position=@($config.test.teleport.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
                health=$config.test.initial_health;armor=0;weapon=$(if($using:Synchronous -and $using:Loadout -eq 'blaster'){'Blaster'}else{'Shotgun'});ammo=$(if($using:Synchronous -and $using:Loadout -eq 'blaster'){0}else{20});enemy_class=$config.test.spawn_class
                enemy_position=@($config.test.spawn_soldier.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
            }|ConvertTo-Json|Set-Content -LiteralPath $resetExpectation -Encoding utf8NoBOM
            $exportArguments=@('--trace',(Join-Path $out 'bot.jsonl'),'--out',$dataset,'--worker',"worker-$worker",'--episode',"seed-$($using:Seed+$index)",'--end-reason','game_frame_limit','--server-log',(Join-Path $out 'server.log'),'--client-name','SoloRetreatBot','--require-execution','--reset-expectation',$resetExpectation)
            if($using:Synchronous){$exportArguments+='--synchronous'}
            if($using:RewardConfig){$exportArguments+=@('--reward-config',$using:RewardConfig)}
            & $using:exporter @exportArguments
            if($LASTEXITCODE){throw 'Transition export failed'}
            $datasetReport=Get-Content -LiteralPath (Join-Path $dataset 'report.json') -Raw|ConvertFrom-Json
            if($using:Synchronous){
                $rngLog=Get-Content -LiteralPath (Join-Path $out 'server.log')
                $rng=@($rngLog|Select-String '^g_test_rng_start game_frame=(\d+) phase=post_frame seed=(\d+) cursor_before=(\d+) cursor_after=256$')
                $rngRelease=@($rngLog|Select-String 'g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
                if($rng.Count -ne 1 -or $rngRelease.Count -ne 1 -or [int]$rng[0].Matches[0].Groups[2].Value -ne ($using:Seed+$index) -or $rng[0].Matches[0].Groups[1].Value -ne $rngRelease[0].Matches[0].Groups[1].Value){throw 'Post-frame RNG seed not confirmed'}
                if($using:ReleaseGameFrame -and [int]$rng[0].Matches[0].Groups[1].Value -ne $using:ReleaseGameFrame){throw 'Fixed release frame differs'}
                if($using:ReleaseGameFrame){
                    $weapon=@($rngLog|Select-String '^g_test_weapon_start game_frame=(\d+) actor=1 weapon=Blaster gunframe_before=\d+ gunframe_after=9$')
                    if($weapon.Count -ne 1 -or [int]$weapon[0].Matches[0].Groups[1].Value -ne $using:ReleaseGameFrame){throw 'Fixed weapon phase differs'}
                    $world=@($rngLog|Select-String '^g_test_world_start frame=(\d+) phase=fixed_map_hold free_pool_reset=1$')
                    if($world.Count -ne 1 -or [int]$world[0].Matches[0].Groups[1].Value -ne $using:ReleaseGameFrame){throw 'Fixed map preparation hold unconfirmed'}
                }
            }
            $curriculumProof=$null
            if($using:TrainingMonsterHealth){
                $nativeLog=Get-Content -LiteralPath (Join-Path $out 'server.log')
                $init=@($nativeLog|Select-String '^g_test_curriculum monster_health map=base1 game_frame=(\d+) entity=(\d+) class=monster_parasite before=175 after=(\d+)$')
                $release=@($nativeLog|Select-String 'g_test_combat_start game_frame=(\d+) ready=1 seed=\d+$')
                if($init.Count -ne 1 -or $release.Count -ne 1 -or [int]$init[0].Matches[0].Groups[3].Value -ne $using:TrainingMonsterHealth -or $init[0].Matches[0].Groups[1].Value -ne $release[0].Matches[0].Groups[1].Value){throw 'Native curriculum initialization not confirmed'}
                $curriculumProof=@{enemy_health=$using:TrainingMonsterHealth;entity=[int]$init[0].Matches[0].Groups[2].Value;game_frame=[int]$init[0].Matches[0].Groups[1].Value;scope='Native test-only initialization at barrier release, not a policy feature'}
            }
            $feedbackAuditReport=$null
            if($using:Feedback){$feedbackAuditReport=& $using:feedbackAudit -RunRoot $out -RelayRoot $feedbackRoot}
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
                curriculum_init=$curriculumProof
                harness_accepted=[bool]$report.accepted;harness_reason=$report.reason;worker_exit_code=$process.ExitCode
                capture_valid=($captures.Count -eq $rows.Count -and $mismatches -eq 0 -and $field.decode_errors -eq 0 -and $field.game_frames -eq $using:GameFrames -and $seedAck -and $dispatchValid -and $datasetReport.command_proof.accepted -and $datasetReport.observed_reset_confirmed -and (!$episodeProvider -or $episodeProviderHash -eq (Get-FileHash -LiteralPath $episodeProvider).Hash))
                provider_config_sha256=$episodeProviderHash
                feedback=$feedbackReport
                feedback_audit=$feedbackAuditReport
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
$results=@($results|Sort-Object worker,episode)
$valid=($fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)))
$valid=$valid -and $nativeFingerprint -eq (Get-HarnessFingerprint (Get-HarnessNativeSourceRecords $nativeRepo))
if($ProviderFile){$valid=$valid -and $probeHash -eq (Get-FileHash -LiteralPath $ProviderFile).Hash}
if($RewardConfig){$valid=$valid -and $rewardHash -eq (Get-FileHash -LiteralPath $RewardConfig).Hash}
$usable=@($results | Where-Object capture_valid)
$manifest=[ordered]@{
    version=2;stage=$(if($ppoProvider){'R4b PPO rollout pilot'}elseif($trainedProvider){'R4a BC control pilot'}else{'R1 dispatch and R2 transition pilot'});provider=$CombatMode;provider_kind=$providerKind;model_weights=$(if($trainedProvider){$ProviderFile}else{$null});model_weights_sha256=$(if($trainedProvider){$probeHash}else{$null});probe_sha256=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{$null});exporter_sha256=(Get-FileHash -LiteralPath $exporter).Hash
    remote_peer=[bool]$remoteProvider
    feedback=[bool]$Feedback;feedback_version=$(if($Feedback){'combat_feedback_v1'}else{$null});relay_sha256=$(if($Feedback){(Get-FileHash -LiteralPath $relay).Hash}else{$null})
    training_monster_health=$TrainingMonsterHealth
    post_frame_rng_reset=[bool]$Synchronous
    release_game_frame=$ReleaseGameFrame
    fixed_world_hold=[bool]$ReleaseGameFrame
    game_frame_budget=$(if($Synchronous){'Commands after combat barrier; preparation excluded'}else{'All commands; preparation included'})
    observation_version='combat_observation_v3';action_version='combat_action_v1';reward_version=$(if($RewardConfig){(Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json).version}else{$null});reward_config_sha256=$(if($RewardConfig){$rewardHash}else{$null});reward_config=$(if($RewardConfig){Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json}else{$null});server_outcome_version=$(if($Synchronous){'server_step_effects_v1'}else{'server_damage_window_v1'})
    timescale=$Timescale;game_frames=$GameFrames;workers=$Workers;episodes_per_worker=$EpisodesPerWorker;loadout=$Loadout;mixed=[bool]$Mixed;health_kit=[bool]$HealthKit;synchronous=[bool]$Synchronous;teacher_vertical=[bool]$TeacherVertical
    reset=$(if($Synchronous){'Cold native server restart per episode; first usable fixture and fresh inventory verified; single-client barrier confirms independent episode RNG seed. Full-world/AI equivalence remains unconfirmed.'}else{'Cold native server restart per episode, verified first usable observed fixture fields. Inventory/RNG/AI/full-world equivalence remain unconfirmed.'})
    source_fingerprint=$fingerprint;provenance_valid=$valid;sources=$sources
    native_source_fingerprint=$nativeFingerprint;native_sources=$nativeSources
    client_sha256=(Get-FileHash -LiteralPath $client).Hash
    plan_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'docs/learned_system1_plan.md')).Hash
    bsp_assets=@(Get-HarnessFileRecords -Root (Join-Path (Split-Path $repo -Parent) 'assets/baseq2') -Paths @((Join-Path (Split-Path $repo -Parent) 'assets/baseq2/pak0.pak')))
    seeds=@($results.seed);physics=$(if($Synchronous){'Stock native PMove and tick; world pauses between synchronous actions, AI frozen during fixture preparation. Live params remain in worker config, without RCON.'}else{'Stock native physics; only timescale and loopback rate differ. Live params remain in worker config, without RCON.'})
    seed_assignments=@($results|Select-Object worker,episode,seed,port)
    stop='Fixed game-frame limit per client; 60s wall watchdog; first-life diagnostics and per-life transitions remain separate.'
    dataset_scope=$(if($RewardConfig){'Separate observations, execution proof, native effects and explicit experimental first-life rewards. No positive demonstration labels or victory inference.'}else{'Separate observed steps, exact native command dispatch proof and damage effect windows; effects are not shot accuracy or delayed causal credit. No scalar reward or positive demonstration labels.'})
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
if(!$KeepRuntimeAssets){
    $artifactBoundary=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts')).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
    foreach($result in $results){
        if([IO.Path]::GetFullPath($result.root).StartsWith($artifactBoundary,[StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath (Join-Path $result.root 'runtime/.q2go-prepared-runtime'))){
            & "$PSScriptRoot/cleanup_completed_runtime_paks.ps1" -RuntimeRoot (Join-Path $result.root 'runtime')
        }
    }
}
if(!$report.capture_complete){throw 'Baseline capture/provenance incomplete; inspect preserved reports'}
