[CmdletBinding()]
param(
    [ValidateRange(1,4)][int]$Workers=4,
    [ValidateRange(1,20)][int]$EpisodesPerWorker=2,
    [ValidateSet(1,2)][int]$Timescale=2,
    [ValidateRange(20,500)][int]$GameFrames=300,
    [int]$Seed=9300,
    [ValidateRange(0,3)][int]$Skill=1,
    [ValidateSet(0,10,20,30,40,60,100)][int]$TrainingMonsterHealth=0,
    [ValidateSet(0,100)][int]$ReleaseGameFrame=0,
    [ValidateRange(1024,65530)][int]$Port=32940,
    [ValidateSet('stocked','blaster','machinegun','weapons','weapons-scarce','shotgun','hyper','rail','scarce')][string]$Loadout='stocked',
    [switch]$Mixed,
    [ValidateSet('uniform','mixed-solo-mixed')][string]$EpisodePattern='uniform',
    [ValidateSet('standard','remaining-far')][string]$SoloFixture='standard',
    [switch]$HealthKit,
    [switch]$Synchronous,
    [switch]$StopOnGoal,
    [switch]$StopOnFirstDeath,
    [switch]$KeepRuntimeAssets,
    [switch]$TeacherVertical,
    [switch]$Feedback,
    [ValidateSet('rules','learned-shadow','learned')][string]$CombatMode='rules',
    [string]$ProviderFile='',
    [string]$RewardConfig='',
    [string]$GeneratedFixtures='',
    [string]$BinaryBundle='',
    [string]$OutputRoot=''
)
$ErrorActionPreference='Stop'
if($Seed -lt 0 -or [long]$Seed+$Workers*$EpisodesPerWorker-1 -gt 2147483647){throw 'Each independent episode needs its own valid 31-bit game seed'}
if($Synchronous -and $Loadout -notin 'blaster','machinegun','shotgun','weapons','weapons-scarce'){throw 'Unsupported synchronous loadout'}
if($Loadout -in 'weapons','weapons-scarce' -and (!$Synchronous -or $CombatMode -ne 'learned' -or $Feedback -or $TrainingMonsterHealth)){throw 'Weapon-choice fixture requires synchronous direct learned control without overrides'}
if($Loadout -eq 'machinegun' -and (!$Synchronous -or $Feedback -or $TrainingMonsterHealth)){throw 'Machinegun requires synchronous capture without health/feedback overrides'}
if($Loadout -eq 'shotgun' -and (!$Synchronous -or $CombatMode -ne 'rules')){throw 'Shotgun exercise requires synchronous rules'}
if($TeacherVertical -and (!$Synchronous -or $CombatMode -ne 'rules' -or $Loadout -ne 'shotgun' -or $Feedback)){throw 'Vertical exercise requires synchronous fixed Shotgun rules without feedback'}
if($RewardConfig -and !$Synchronous){throw 'Reward export requires -Synchronous'}
if($TrainingMonsterHealth -and (!$Synchronous -or $Loadout -ne 'blaster' -or $Mixed -or $HealthKit -or $Feedback)){throw 'Curriculum requires isolated synchronous Blaster fixture'}
if($ReleaseGameFrame -and (!$Synchronous -or $Loadout -notin 'blaster','machinegun','weapons','weapons-scarce' -or $HealthKit -or $GameFrames -lt 150)){throw 'Fixed release requires a supported synchronous fixture without health kit and >=150 game frames'}
if($EpisodePattern -ne 'uniform' -and (!$Mixed -or $EpisodesPerWorker -ne 3 -or !$Synchronous -or $Loadout -notin 'blaster','machinegun','weapons','weapons-scarce' -or $CombatMode -ne 'learned' -or $Feedback -or $HealthKit -or $TrainingMonsterHealth)){throw 'Unsupported Mixed/Solo pattern'}
if($SoloFixture -ne 'standard' -and (!$Synchronous -or $Loadout -ne 'blaster' -or $CombatMode -ne 'learned' -or $Feedback -or $HealthKit -or $TrainingMonsterHealth -or ($Mixed -and $EpisodePattern -eq 'uniform'))){throw 'Remaining-Parasite fixture requires isolated direct synchronous Blaster Solo episodes without health overrides'}
if($StopOnGoal -and (!$Synchronous -or $Feedback)){throw 'Goal stop requires synchronous capture without a live feedback relay'}
if($StopOnFirstDeath -and (!$StopOnGoal -or !$RewardConfig)){throw 'First-life death stop requires goal supervision and an explicit reward'}
$repo=Split-Path $PSScriptRoot -Parent
if($GeneratedFixtures){
    if(!$Synchronous -or $ReleaseGameFrame -ne 100 -or $Feedback -or $HealthKit -or $TrainingMonsterHealth -or $SoloFixture -ne 'standard' -or $EpisodePattern -ne 'uniform'){throw 'Generated fixtures require the standard synchronous fixed release without overrides'}
    $GeneratedFixtures=(Resolve-Path -LiteralPath $GeneratedFixtures).Path
    $sampled=Get-Content -LiteralPath $GeneratedFixtures -Raw|ConvertFrom-Json
    if($sampled.instances.Count -ne $Workers*$EpisodesPerWorker){throw 'Wrong generated fixture count'}
    . "$PSScriptRoot/generated_combat_fixture.ps1"
    for($i=0;$i -lt $sampled.instances.Count;$i++){
        $temp=$sampled.instances[$i]
        if($temp.engine_seed -ne $Seed+$i -or $temp.loadout -ne $Loadout -or $temp.skill -ne $Skill){throw 'Generated conditions differ from batch'}
    }
}
. "$PSScriptRoot/harness_manifest.ps1"
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/learned-combat-baseline-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output directory required'}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$OutputRoot=(Resolve-Path -LiteralPath $OutputRoot).Path
if($CombatMode -ne 'rules' -and $Loadout -notin 'blaster','machinegun','weapons','weapons-scarce'){throw 'Unsupported direct/shadow loadout'}
if($CombatMode -ne 'rules' -and !$ProviderFile){$ProviderFile=Join-Path $PSScriptRoot 'scenarios/combat-control-probe.json'}
if($ProviderFile){$ProviderFile=(Resolve-Path -LiteralPath $ProviderFile).Path}
$remoteProvider=$false
$providerKind=$null
if($ProviderFile){$providerKind=(Get-Content -LiteralPath $ProviderFile -Raw|ConvertFrom-Json).kind;$remoteProvider=($providerKind -eq 'combat_remote_v1')}
$ppoProvider=($providerKind -eq 'combat_ppo_v1')
if($Loadout -in 'weapons','weapons-scarce' -and (!$ppoProvider -or (Get-Content -LiteralPath $ProviderFile -Raw|ConvertFrom-Json).weapon_head -ne 'combat_masked_weapon_v1')){throw 'Weapon-choice fixture requires a versioned PPO weapon head'}
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
$exporter=Join-Path $OutputRoot 'q2combat-export.exe'
if($BinaryBundle){
    if($Feedback){throw 'Feedback relay does not yet support prebuilt bundles'}
    . "$PSScriptRoot/combat_harness_bundle.ps1"
    $bundleReceipt=Assert-HarnessBinaryBundle $BinaryBundle $fingerprint
    foreach($name in @('q2coopbot.exe','q2combat-export.exe')){
        Install-HarnessBinaryLink (Join-Path $BinaryBundle $name) (Join-Path $OutputRoot $name) $bundleReceipt.binaries.$name.sha256
    }
}else{
$env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
$env:GOTOOLCHAIN='auto'
Push-Location $repo
try{go build -buildvcs=false -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
Push-Location $repo;try{go build -buildvcs=false -o $exporter ./cmd/q2combat-export;if($LASTEXITCODE){throw 'Exporter build failed'}}finally{Pop-Location}
}
$relay=Join-Path $OutputRoot 'q2learning-relay.exe'
if($Feedback){Push-Location $repo;try{go build -buildvcs=false -o $relay ./cmd/q2learning-relay;if($LASTEXITCODE){throw 'Relay build failed'}}finally{Pop-Location}}
$probeHash=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{''})
$generatedHash=$(if($GeneratedFixtures){(Get-FileHash -LiteralPath $GeneratedFixtures).Hash}else{''})
$hostExe=(Get-Process -Id $PID).Path
$runner=Join-Path $PSScriptRoot 'run_solo_tactical_retreat.ps1'
$feedbackAudit=Join-Path $PSScriptRoot 'audit_learning_feedback.ps1'
$clock=[Diagnostics.Stopwatch]::StartNew()
$results=@(0..($Workers-1) | ForEach-Object -Parallel {
    $ErrorActionPreference='Stop'
    $worker=$_
    for($episode=0;$episode -lt $using:EpisodesPerWorker;$episode++){
        $index=$worker*$using:EpisodesPerWorker+$episode
        $episodeMixed=[bool]$using:Mixed -and ($using:EpisodePattern -eq 'uniform' -or $episode -ne 1)
        $episodeFixture=if($episodeMixed){'standard'}else{$using:SoloFixture}
        $out=Join-Path $using:OutputRoot "worker-$worker-episode-$episode"
        $sample=$null;$samplePath=''
		$captureMap='base1'
        if($using:GeneratedFixtures){
            . "$using:repo/scripts/generated_combat_fixture.ps1"
            $sample=(Get-Content -LiteralPath $using:GeneratedFixtures -Raw|ConvertFrom-Json).instances[$index]
			if($sample.map){$captureMap=$sample.map}
            $samplePath=Join-Path $using:OutputRoot "fixture-$index.json"
            $sample|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $samplePath -Encoding utf8NoBOM
            $null=Read-GeneratedCombatFixture $samplePath ($using:Seed+$index) $using:Loadout $episodeMixed
        }
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
        $arguments+=@('-RecoverySkill',$using:Skill)
        $arguments+=@('-ParasiteFixture',$episodeFixture)
        if($samplePath){$arguments+=@('-GeneratedFixture',$samplePath)}
        if($using:Synchronous){$arguments+='-Synchronous'};if($using:StopOnGoal){$arguments+='-StopOnGoal'};if($using:StopOnFirstDeath){$arguments+='-StopOnFirstDeath'}
        if($using:TeacherVertical){$arguments+='-TeacherVertical'}
        if($episodeMixed){$arguments+=@('-ParasiteMixed','-ParasiteMixedClass','monster_gunner')}
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
            $goal=$null;$goalValid=$false
            if($using:StopOnGoal -and (Test-Path (Join-Path $out 'goal-stop.json'))){
                . "$using:repo/scripts/combat_goal_stop.ps1"
                . "$using:repo/scripts/read_damage_events.ps1"
                $goal=Get-Content (Join-Path $out 'goal-stop.json') -Raw | ConvertFrom-Json
                $release=@(Get-Content (Join-Path $out 'server.log') | Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
                if($release.Count -ne 1){throw 'Goal release proof missing'}
                $classes=if($sample){@($sample.monsters.class)}else{@('monster_parasite')};if(!$sample -and $episodeMixed){$classes+='monster_gunner'}
                $verified=Get-CombatGoalReceipt $rows[-1] @(Read-DamageEvents (Join-Path $out 'server.log')) @{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value} $classes $captureMap
                $goalValid=$verified -and $goal.reason -eq 'combat_goal_complete' -and $goal.kill_frame -eq $verified.kill_frame -and $goal.observed_frame -ge $verified.kill_frame+1 -and $goal.observed_frame -le $rows[-1].observation_frame -and $goal.spawncount -eq $verified.spawncount -and $goal.actor -eq $verified.actor -and (Test-Path (Join-Path $out 'goal.stop'))
                if(!$goalValid){throw 'Unverified goal stop'}
            }
            $death=$null;$deathValid=$false
            if($using:StopOnFirstDeath -and (Test-Path (Join-Path $out 'death-stop.json'))){
                if($goalValid){throw 'Conflicting goal and death receipts'}
                . "$using:repo/scripts/combat_death_stop.ps1"
                . "$using:repo/scripts/read_damage_events.ps1"
                $death=Get-Content (Join-Path $out 'death-stop.json') -Raw | ConvertFrom-Json
                $release=@(Get-Content (Join-Path $out 'server.log') | Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
                if($release.Count -ne 1){throw 'Death release proof missing'}
                $observed=@($rows|Where-Object observation_frame -eq $death.observed_frame)
                if($observed.Count -ne 1){throw 'Death receipt observation missing'}
                $verified=Get-CombatFirstLifeDeathReceipt $observed[0] @(Read-DamageEvents (Join-Path $out 'server.log')) @{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value} $captureMap
                if(!$verified){throw 'Unverified death stop'}
                foreach($key in @('version','reason','map','spawncount','actor','death_frame','observed_frame','health')){if($death.$key -ne $verified.$key){throw 'Death receipt differs from native evidence'}}
                if(!(Test-Path (Join-Path $out 'goal.stop')) -or (Get-Content (Join-Path $out 'goal.stop') -Raw).Trim() -ne 'combat_first_life_death'){throw 'Death stop marker missing'}
                $deathValid=$true
            }
            $frameBudgetValid=$field.game_frames -eq $using:GameFrames -or (($goalValid -or $deathValid) -and $field.game_frames -gt 0 -and $field.game_frames -lt $using:GameFrames)
            $dataset=Join-Path $out 'dataset'
            $config=Get-Content -LiteralPath (Join-Path $out 'bot-config.json') -Raw|ConvertFrom-Json
            $generatedProof=$null
            if($sample){$generatedProof=Confirm-GeneratedCombatStart $sample (Join-Path $out 'server.log')}
            if($episodeFixture -eq 'remaining-far' -and ($config.test.teleport -ne '-48,16,24' -or $config.test.spawn_soldier -ne '200,-224,24' -or $config.test.initial_health -ne 100 -or $config.test.spawn_class -ne 'monster_parasite')){throw 'Remaining-Parasite config differs from declared fixture'}
            $resetExpectation=Join-Path $out 'reset-expectation.json'
            @{version='observed_fixture_reset_v1';map=$config.test.teleport_map;seed=$(if($using:Synchronous){$using:Seed+$index}else{$null})
                position=@($config.test.teleport.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
                health=$config.test.initial_health;armor=0;weapon=$(if($using:Loadout -in 'machinegun','weapons','weapons-scarce'){'Machinegun'}elseif($using:Synchronous -and $using:Loadout -eq 'blaster'){'Blaster'}else{'Shotgun'});ammo=$(switch($using:Loadout){machinegun{100};weapons{40};weapons-scarce{10};blaster{if($using:Synchronous){0}else{20}};default{20}});enemy_class=$config.test.spawn_class
                inventory=$(if($using:Loadout -in 'weapons','weapons-scarce'){@(@{name='Blaster';count=1},@{name='Machinegun';count=1},@{name='Shotgun';count=1},@{name='Bullets';count=$(if($using:Loadout -eq 'weapons'){40}else{10})},@{name='Shells';count=$(if($using:Loadout -eq 'weapons'){20}else{6})})}else{$null})
                enemy_position=@($config.test.spawn_soldier.Split(',')|ForEach-Object {[double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture)})
            }|ConvertTo-Json|Set-Content -LiteralPath $resetExpectation -Encoding utf8NoBOM
            $exportArguments=@('--trace',(Join-Path $out 'bot.jsonl'),'--out',$dataset,'--worker',"worker-$worker",'--episode',"seed-$($using:Seed+$index)",'--end-reason',$(if($goalValid){'combat_goal_complete'}elseif($deathValid){'combat_first_life_death'}else{'game_frame_limit'}),'--server-log',(Join-Path $out 'server.log'),'--client-name','SoloRetreatBot','--require-execution','--reset-expectation',$resetExpectation)
            if($using:Synchronous){$exportArguments+='--synchronous'}
            if($goalValid){$exportArguments+=@('--goal-observed-frame',"$($goal.observed_frame)")}
            if($deathValid){$exportArguments+=@('--death-stop',(Join-Path $out 'death-stop.json'))}
            if($using:RewardConfig){$exportArguments+=@('--reward-config',$using:RewardConfig)}
            & $using:exporter @exportArguments
            if($LASTEXITCODE){throw 'Transition export failed'}
            $datasetReport=Get-Content -LiteralPath (Join-Path $dataset 'report.json') -Raw|ConvertFrom-Json
            if($deathValid -and ((Get-Content (Join-Path $dataset 'death-stop-verification.json') -Raw|ConvertFrom-Json).state -ne 'verified')){throw 'Death terminal/reward verification missing'}
            if($using:Synchronous){
                $rngLog=Get-Content -LiteralPath (Join-Path $out 'server.log')
                $rng=@($rngLog|Select-String '^g_test_rng_start game_frame=(\d+) phase=post_frame seed=(\d+) cursor_before=(\d+) cursor_after=256$')
                $rngRelease=@($rngLog|Select-String 'g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$')
                if($rng.Count -ne 1 -or $rngRelease.Count -ne 1 -or [int]$rng[0].Matches[0].Groups[2].Value -ne ($using:Seed+$index) -or $rng[0].Matches[0].Groups[1].Value -ne $rngRelease[0].Matches[0].Groups[1].Value){throw 'Post-frame RNG seed not confirmed'}
                if($using:ReleaseGameFrame -and [int]$rng[0].Matches[0].Groups[1].Value -ne $using:ReleaseGameFrame){throw 'Fixed release frame differs'}
                if($using:ReleaseGameFrame){
                    $skillProof=@($rngLog|Select-String '^g_test_skill_start game_frame=(\d+) skill=(\d+)$')
                    if($skillProof.Count -ne 1 -or [int]$skillProof[0].Matches[0].Groups[1].Value -ne $using:ReleaseGameFrame -or [int]$skillProof[0].Matches[0].Groups[2].Value -ne $using:Skill){throw 'Native skill not confirmed at release'}
                    $expectedWeaponPhase=if($using:Loadout -in 'machinegun','weapons','weapons-scarce'){'Machinegun gunframe_before=\d+ gunframe_after=6'}else{'Blaster gunframe_before=\d+ gunframe_after=9'}
                    $weapon=@($rngLog|Select-String ("^g_test_weapon_start game_frame=(\d+) actor=1 weapon="+$expectedWeaponPhase+'$'))
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
            # The extra Gunner is in the fixture entity lump; verify the actual native input.
            $captureMap=if($sample -and $sample.map){$sample.map}else{'base1'}
            $entities=Get-Content -LiteralPath (Join-Path $out "runtime/baseq2/maps/$captureMap.ent") -Raw
            $gunnerCount=[regex]::Matches($entities,'"classname"\s+"monster_gunner"').Count
            if($gunnerCount -ne [int]$episodeMixed){throw 'Episode composition differs from declared pattern'}
            $life=Get-Content -LiteralPath (Join-Path $out 'combat-first-life.json') -Raw | ConvertFrom-Json
            $files=@('q2ded.exe','baseq2/game.dll',"baseq2/maps/$captureMap.ent","baseq2/maps/$captureMap.aas") | ForEach-Object {
                $path=Join-Path (Join-Path $out 'runtime') $_
                if(Test-Path -LiteralPath $path){@{path=$_;sha256=(Get-FileHash -LiteralPath $path).Hash}}
            }
            [pscustomobject]@{
                worker=$worker;episode=$episode;seed=($using:Seed+$index);port=($using:Port+$worker);seed_confirmed=$seedAck
                curriculum_init=$curriculumProof;fixture_mixed=$episodeMixed;fixture_gunner_count=$gunnerCount;solo_fixture=$episodeFixture
                generated_fixture=$sample;generated_start=$generatedProof
                harness_accepted=[bool]$report.accepted;harness_reason=$report.reason;worker_exit_code=$process.ExitCode
                capture_valid=($captures.Count -eq $rows.Count -and $mismatches -eq 0 -and $field.decode_errors -eq 0 -and $frameBudgetValid -and $seedAck -and $dispatchValid -and $datasetReport.command_proof.accepted -and $datasetReport.observed_reset_confirmed -and (!$episodeProvider -or $episodeProviderHash -eq (Get-FileHash -LiteralPath $episodeProvider).Hash))
                goal_stop=$goal;death_stop=$death;actual_game_frames=$field.game_frames;frame_budget_valid=[bool]$frameBudgetValid
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
if($GeneratedFixtures){$valid=$valid -and $generatedHash -eq (Get-FileHash -LiteralPath $GeneratedFixtures).Hash}
$usable=@($results | Where-Object capture_valid)
$manifest=[ordered]@{
    version=2;stage=$(if($ppoProvider){'R4b PPO rollout pilot'}elseif($trainedProvider){'R4a BC control pilot'}else{'R1 dispatch and R2 transition pilot'});provider=$CombatMode;provider_kind=$providerKind;model_weights=$(if($trainedProvider){$ProviderFile}else{$null});model_weights_sha256=$(if($trainedProvider){$probeHash}else{$null});probe_sha256=$(if($ProviderFile){(Get-FileHash -LiteralPath $ProviderFile).Hash}else{$null});exporter_sha256=(Get-FileHash -LiteralPath $exporter).Hash
    remote_peer=[bool]$remoteProvider
    feedback=[bool]$Feedback;feedback_version=$(if($Feedback){'combat_feedback_v1'}else{$null});relay_sha256=$(if($Feedback){(Get-FileHash -LiteralPath $relay).Hash}else{$null})
    training_monster_health=$TrainingMonsterHealth
    generated_fixtures_sha256=$generatedHash
    post_frame_rng_reset=[bool]$Synchronous
    release_game_frame=$ReleaseGameFrame
    fixed_world_hold=[bool]$ReleaseGameFrame
    stop_on_goal=[bool]$StopOnGoal
    monster_no_infighting=[bool]$Synchronous
    standard_monster_spawn_height=24.125
    game_frame_budget=$(if($Synchronous){'Commands after combat barrier; preparation excluded'}else{'All commands; preparation included'})
    observation_version='combat_observation_v3';action_version='combat_action_v1';reward_version=$(if($RewardConfig){(Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json).version}else{$null});reward_config_sha256=$(if($RewardConfig){$rewardHash}else{$null});reward_config=$(if($RewardConfig){Get-Content -LiteralPath $RewardConfig -Raw|ConvertFrom-Json}else{$null});server_outcome_version=$(if($Synchronous){'server_step_effects_v1'}else{'server_damage_window_v1'})
    timescale=$Timescale;game_frames=$GameFrames;workers=$Workers;episodes_per_worker=$EpisodesPerWorker;loadout=$Loadout;skill=$Skill;mixed=[bool]$Mixed;episode_pattern=$EpisodePattern;solo_fixture=$SoloFixture;health_kit=[bool]$HealthKit;synchronous=[bool]$Synchronous;teacher_vertical=[bool]$TeacherVertical
    reset=$(if($Synchronous){'Cold native server restart per episode; first usable fixture and fresh inventory verified; single-client barrier confirms independent episode RNG seed. Full-world/AI equivalence remains unconfirmed.'}else{'Cold native server restart per episode, verified first usable observed fixture fields. Inventory/RNG/AI/full-world equivalence remain unconfirmed.'})
    source_fingerprint=$fingerprint;provenance_valid=$valid;sources=$sources
    native_source_fingerprint=$nativeFingerprint;native_sources=$nativeSources
    client_sha256=(Get-FileHash -LiteralPath $client).Hash
    plan_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'docs/learned_system1_plan.md')).Hash
    bsp_assets=@(Get-HarnessFileRecords -Root (Join-Path (Split-Path $repo -Parent) 'assets/baseq2') -Paths @((Join-Path (Split-Path $repo -Parent) 'assets/baseq2/pak0.pak')))
    seeds=@($results.seed);physics=$(if($Synchronous){'Stock native PMove and tick; world pauses between synchronous actions, AI frozen during fixture preparation. Live params remain in worker config, without RCON.'}else{'Stock native physics; only timescale and loopback rate differ. Live params remain in worker config, without RCON.'})
    seed_assignments=@($results|Select-Object worker,episode,seed,port,fixture_mixed,solo_fixture)
    stop=$(if($StopOnGoal){'Verified native fixture kills in first life, followed by observed final effect; otherwise fixed game-frame maximum and wall watchdog.'}else{'Fixed game-frame limit per client; scenario wall watchdog; first-life diagnostics and per-life transitions remain separate.'})
    dataset_scope=$(if($RewardConfig){'Separate observations, execution proof, native effects and explicit experimental first-life rewards. No positive demonstration labels or victory inference.'}else{'Separate observed steps, exact native command dispatch proof and damage effect windows; effects are not shot accuracy or delayed causal credit. No scalar reward or positive demonstration labels.'})
}
if($BinaryBundle){$manifest.binary_bundle=$BinaryBundle;$manifest.binary_bundle_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $BinaryBundle 'receipt.json')).Hash.ToLowerInvariant()}
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
$streamBoundary=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts')).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
if([IO.Path]::GetFullPath($OutputRoot).StartsWith($streamBoundary,[StringComparison]::OrdinalIgnoreCase)){
    & "$PSScriptRoot/compress_completed_combat_streams.ps1" -Root $OutputRoot
}
if(!$report.capture_complete){throw 'Baseline capture/provenance incomplete; inspect preserved reports'}
