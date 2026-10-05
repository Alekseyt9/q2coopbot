[CmdletBinding()]
param([ValidateSet(0,100)][int]$ReleaseGameFrame=0,[ValidateSet(0,10,20,30,40,60,100)][int]$TrainingMonsterHealth=0,[switch]$TeacherVertical,[switch]$Synchronous,[switch]$Worker,[ValidateRange(0,500)][int]$GameFrames=0,[ValidateSet('rules','learned-shadow','learned')][string]$CombatMode='rules',[string]$ProviderFile='',[switch]$Rules,[switch]$CombatCapture,[ValidateSet(1,2)][int]$Timescale=2,[switch]$ParasiteWeapon,[switch]$ParasiteMixed,[switch]$RequireMixedDetour,[ValidateSet('monster_infantry','monster_gunner')][string]$ParasiteMixedClass='monster_infantry',[switch]$ParasiteHealthKit,[ValidateRange(1,100)][int]$ParasiteHealth=100,[ValidateSet('stocked','blaster','shotgun','hyper','rail','scarce')][string]$ParasiteLoadout='stocked',[switch]$CornerEscape,[switch]$Recovery,[ValidateRange(0,3)][int]$RecoverySkill=1,[ValidateRange(1,100)][int]$RecoveryHealth=55,[switch]$Group,[switch]$GroupRetreat,[switch]$Circle,[switch]$Cover,[switch]$CoverFight,[int]$CoverTargetX=200,[int]$Seed=601,[int]$Port=31820,[string]$OutputRoot='',[string]$Client='',[string]$System1='hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0')
$ErrorActionPreference='Stop';$repo=Split-Path $PSScriptRoot -Parent
function Measure-BarrelSafety($Rows,$Events) {
    $actor=$Rows[0].self_entity
    $blocked=@($Rows|Where-Object {$_.arbitration.limit_reason -eq 'barrel_blast_risk'})
    @{
        observed_frames=@($Rows|Where-Object {$_.barrels.Count -gt 0}).Count
        suppressed_frames=$blocked.Count
        suppressed_attack_violations=@($blocked|Where-Object {$_.sent_command.Buttons -band 1}).Count
        self_health_damage=[int](($Events|Where-Object {$_.target -eq $actor -and $_.mod -eq 26}|Measure-Object live_health_damage -Sum).Sum)
        barrel_contacts=@($Events|Where-Object {$_.attacker -eq $actor -and $_.target_class -eq 'misc_explobox' -and $_.mod -ne 26}).Count
        scope='Observed stock barrels and Blaster shot guard; no unseen barrel or other weapon acceptance'
    }
}
if($CoverFight){$Cover=$true}
if($Group){$Circle=$true}
if($GroupRetreat -and ($Group -or $Circle -or $Cover)){throw 'GroupRetreat is a separate fixture'}
if($Recovery -and ($Group -or $GroupRetreat -or $Circle -or $Cover)){throw 'Recovery is a separate fixture'}
if($Circle -and $Cover){throw 'Circle and Cover are separate fixtures'}
if($CornerEscape -and ($Recovery -or $Group -or $GroupRetreat -or $Circle -or $Cover)){throw 'CornerEscape is a separate fixture'}
if($ParasiteWeapon -and ($Recovery -or $CornerEscape -or $Group -or $GroupRetreat -or $Circle -or $Cover)){throw 'ParasiteWeapon is a separate fixture'}
if(($ParasiteMixed -or $ParasiteHealthKit) -and !$ParasiteWeapon){throw 'Parasite variants require ParasiteWeapon'}
if($RequireMixedDetour -and (!$ParasiteMixed -or $ParasiteMixedClass -ne 'monster_gunner')){throw 'Required mixed detour needs observed Gunner/Parasite fixture'}
if($Synchronous -and (!$Rules -or !$CombatCapture -or !$ParasiteWeapon -or $ParasiteLoadout -notin 'blaster','shotgun')){throw 'Synchronous learning requires Rules, CombatCapture and supported ParasiteWeapon equipment'}
if($ParasiteLoadout -eq 'shotgun' -and (!$Synchronous -or $CombatMode -ne 'rules')){throw 'Fixed Shotgun exercise requires synchronous rules'}
if(!$Worker){
    . "$PSScriptRoot/harness_manifest.ps1"
    $fingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
    $OutputRoot=Join-Path $repo ('workspace/artifacts/solo-tactical-retreat-'+(Get-Date -Format yyyyMMdd-HHmmss-fff)+'-'+[guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory $OutputRoot|Out-Null
    $Client=Join-Path $OutputRoot 'q2coopbot.exe'
    Push-Location $repo;try{go build -o $Client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Client build failed'}}finally{Pop-Location}
    $hostExe=(Get-Process -Id $PID).Path;$script=$PSCommandPath
    $results=@(0..1|ForEach-Object -Parallel {
        $out=Join-Path $using:OutputRoot "run-$_"
        $args=@('-NoProfile','-File',$using:script,'-Worker','-Seed',($using:Seed+$_),'-Port',($using:Port+$_),'-OutputRoot',$out,'-Client',$using:Client,'-System1',$using:System1)
        $args+=@('-Timescale',$using:Timescale); if($using:Rules){$args+='-Rules'};if($using:CombatCapture){$args+='-CombatCapture'}
        $args+=@('-CombatMode',$using:CombatMode);if($using:ProviderFile){$args+=@('-ProviderFile',$using:ProviderFile)}
        $args+=@('-GameFrames',$using:GameFrames)
        $args+=@('-TrainingMonsterHealth',$using:TrainingMonsterHealth);$args+=@('-ReleaseGameFrame',$using:ReleaseGameFrame)
        if($using:Synchronous){$args+='-Synchronous'}
        if($using:RequireMixedDetour){$args+='-RequireMixedDetour'}
        if($using:Cover){$args+='-Cover'}
        if($using:CoverFight){$args+='-CoverFight'}
        if($using:Circle){$args+='-Circle'}
        if($using:Group){$args+='-Group'}
        if($using:GroupRetreat){$args+='-GroupRetreat'}
        if($using:ParasiteWeapon){$args+=@('-ParasiteWeapon','-ParasiteLoadout',$using:ParasiteLoadout,'-RecoverySkill',$using:RecoverySkill,'-ParasiteHealth',$using:ParasiteHealth);if($using:ParasiteMixed){$args+=@('-ParasiteMixed','-ParasiteMixedClass',$using:ParasiteMixedClass)};if($using:ParasiteHealthKit){$args+='-ParasiteHealthKit'}}
        if($using:CornerEscape){$args+=@('-CornerEscape','-RecoverySkill',$using:RecoverySkill)}
        if($using:Recovery){$args+=@('-Recovery','-RecoverySkill',$using:RecoverySkill,'-RecoveryHealth',$using:RecoveryHealth)}
        $args+=@('-CoverTargetX',$using:CoverTargetX)
        $child=Start-Process $using:hostExe -ArgumentList @($args|ForEach-Object {'"'+$_+'"'}) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $using:OutputRoot "worker-$_.log") -RedirectStandardError (Join-Path $using:OutputRoot "worker-$_.err")
        $child.WaitForExit();$r=Get-Content (Join-Path $out 'report.json') -Raw|ConvertFrom-Json;if($child.ExitCode){$r.accepted=$false};$r
    } -ThrottleLimit 2)
    $valid=$fingerprint -eq (Get-HarnessFingerprint (Get-HarnessSourceRecords $repo))
    $accepted=$valid -and $results.Count -eq 2 -and @($results|Where-Object {!$_.accepted}).Count -eq 0
    @{accepted=$accepted;provenance_valid=$valid;source_fingerprint=$fingerprint;parasite_weapon=[bool]$ParasiteWeapon;parasite_mixed=[bool]$ParasiteMixed;require_mixed_detour=[bool]$RequireMixedDetour;parasite_mixed_class=$ParasiteMixedClass;parasite_health=$ParasiteHealth;parasite_health_kit=[bool]$ParasiteHealthKit;parasite_loadout=$ParasiteLoadout;corner_escape=[bool]$CornerEscape;recovery=[bool]$Recovery;group=[bool]$Group;group_retreat=[bool]$GroupRetreat;circle=[bool]$Circle;cover=[bool]$Cover;cover_fight=[bool]$CoverFight;cover_target_x=$CoverTargetX;model=$System1;timescale=$Timescale;parallelism=2;rules=[bool]$Rules;combat_capture=[bool]$CombatCapture;seeds=@($Seed,($Seed+1));results=$results}|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json')
    "Solo tactical retreat: $OutputRoot";if(!$accepted){throw 'Solo tactical retreat rejected'};return
}
if($TrainingMonsterHealth -and (!$Synchronous -or !$ParasiteWeapon -or $ParasiteLoadout -ne 'blaster' -or $ParasiteMixed -or $ParasiteHealthKit)){throw 'Unsupported curriculum fixture'}
if($Rules){$System1=''}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Port occupied'}
if(Test-Path $OutputRoot){throw 'Fresh output required'}
New-Item -ItemType Directory $OutputRoot|Out-Null
$resetClock=[Diagnostics.Stopwatch]::StartNew()
$runtime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1 -RuntimeRoot (Join-Path $OutputRoot 'runtime')
if($Recovery){
    # A genuine native 25HP item, introduced before server startup only.
    [IO.File]::AppendAllText((Join-Path $runtime 'baseq2/maps/base1.ent'),"`n{`n`"classname`" `"item_health_large`"`n`"origin`" `"32 -352 24`"`n}`n",[Text.Encoding]::ASCII)
}
if($ParasiteHealthKit){
    [IO.File]::AppendAllText((Join-Path $runtime 'baseq2/maps/base1.ent'),"`n{`n`"classname`" `"item_health_large`"`n`"origin`" `"32 -424 24`"`n}`n",[Text.Encoding]::ASCII)
}
if($Group -or $GroupRetreat -or $ParasiteMixed){
    # The second vulnerable native actor is a startup fixture, not a later
    # gameplay intervention. Original BSP/AAS and mover physics are unchanged.
    $entityPath=Join-Path $runtime 'baseq2/maps/base1.ent'
    $flankOrigin=if($ParasiteMixed){'96 -200 24'}elseif($GroupRetreat){'32 -352 24'}else{'200 -320 24'}
    $flankClass=if($ParasiteMixed){$ParasiteMixedClass}else{'monster_infantry'}
    [IO.File]::AppendAllText($entityPath,"`n{`n`"classname`" `"$flankClass`"`n`"origin`" `"$flankOrigin`"`n}`n",[Text.Encoding]::ASCII)
}
$server=$null;$bot=$null;$trace=Join-Path $OutputRoot 'bot.jsonl';$report=@{accepted=$false;reason='not_run';seed=$Seed}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale $Timescale +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_damage 1 +set g_test_seed $Seed +map base1"
    if($CombatCapture -and !$Synchronous){$args=$args.Replace('+map base1','+set sv_test_trace_client SoloRetreatBot +map base1')}
    if($Synchronous){
        @("set sv_harness_instance learning-$Seed",'set sv_test_trace_client SoloRetreatBot','set sv_test_lockstep_client SoloRetreatBot','set g_test_combat_barrier 1','set g_test_combat_clients 1',"set g_test_combat_monster_health $TrainingMonsterHealth","set g_test_combat_release_frame $ReleaseGameFrame")|Set-Content -LiteralPath (Join-Path $runtime 'baseq2/learning-test.cfg') -Encoding ascii
        $args=$args.Replace('+map base1','+exec learning-test.cfg +map base1')
    }
    if($Recovery -or $CornerEscape -or $ParasiteWeapon){$args=$args.Replace('+map base1',"+set skill $RecoverySkill +map base1");$report.skill=$RecoverySkill;$report.initial_health=$(if($CornerEscape){65}elseif($ParasiteWeapon){$ParasiteHealth}else{$RecoveryHealth})}
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    $deadline=(Get-Date).AddSeconds(15)
    do{Start-Sleep -Milliseconds 100;if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!(Get-NetUDPEndpoint -OwningProcess $server.Id -LocalPort $Port -ErrorAction SilentlyContinue))
    if(Get-NetUDPEndpoint -OwningProcess $server.Id|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback'}
    $report.reset_to_udp_ready_seconds=$resetClock.Elapsed.TotalSeconds
    $report.reset_scope='Cold runtime preparation plus native server restart to UDP bind; protocol begin is measured separately by the client'
    $config=Join-Path $OutputRoot 'bot-config.json'
    $placement=if($ParasiteWeapon -and $ParasiteLoadout -eq 'rail'){'32,-352,24.125'}elseif($CornerEscape){'-40.375,-426,24.125'}elseif($Cover){'240,-416,24.125'}elseif($GroupRetreat){'128,-304,24'}else{'32,-224,24'}
    $enemyClass=if($Cover -or $Circle -or $GroupRetreat){'monster_infantry'}else{'monster_parasite'}
    $enemyOrigin=if($ParasiteWeapon -and $ParasiteLoadout -eq 'rail'){'240,-160,24'}elseif($CornerEscape){'67.125,-316.875,24'}elseif($Cover){"$CoverTargetX,-224,24"}elseif($GroupRetreat){'192,-304,24'}else{'200,-224,24'}
    $initialHealth=if($ParasiteWeapon){$ParasiteHealth}elseif($CornerEscape){65}elseif($Recovery){$RecoveryHealth}elseif($Group -or $GroupRetreat){100}elseif($Cover -or $Circle){25}else{0}
    @{server=@{host='127.0.0.1';port=$Port};client=@{name='SoloRetreatBot';game_dir=(Join-Path $runtime 'baseq2')};models=@{system1=$System1};combat=@{mode=$CombatMode;provider_file=$ProviderFile};run=@{duration=$(if($GameFrames){'60s'}else{'15s'});game_frames=$GameFrames;frame_paced=$true;mode='campaign';next_map='base2'};test=@{teleport_map='base1';teleport=$placement;spawn_map='base1';spawn_soldier=$enemyOrigin;spawn_class=$enemyClass;teacher_vertical=[bool]$TeacherVertical;synchronous=[bool]$Synchronous;combat_barrier=[bool]$Synchronous;setup_hold_frames=$(if($Cover -or $Circle -or $GroupRetreat -or $Recovery -or $CornerEscape -or $ParasiteWeapon){10}else{0});initial_health=$initialHealth;weapon_switch_fixture=$(if($ParasiteWeapon){"parasite_$ParasiteLoadout"}else{""})};output=@{trace_jsonl=$trace;combat_capture=[bool]$CombatCapture}}|ConvertTo-Json -Depth 6|Set-Content $config
    $bot=Start-Process $Client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $waitMilliseconds=$(if($GameFrames){[int]([math]::Max(25,$GameFrames/(10*$Timescale)+10)*1000)}else{25000});$null=$bot.WaitForExit($waitMilliseconds);if(!$bot.HasExited){throw 'Bot timeout'};if($bot.ExitCode){throw 'Bot failed'}
    $rows=@(Get-Content $trace|ForEach-Object {$_|ConvertFrom-Json})
    . "$PSScriptRoot/read_damage_events.ps1"
    $events=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log'))
    $report.barrel_safety=Measure-BarrelSafety $rows $events
    if($report.barrel_safety.suppressed_attack_violations){throw 'Barrel guard retained attack'}
    $report.campaign_engagement=@{
        engage_frames=@($rows|Where-Object {$_.arbitration.combat_intent.action -eq 'engage'}).Count
        pending_hold_frames=@($rows|Where-Object {$_.arbitration.move_limit_reason -eq 'campaign_combat_hold'}).Count
        engage_route_violations=@($rows|Where-Object {$_.arbitration.combat_intent.action -eq 'engage' -and $_.arbitration.move_source -like 'route*'}).Count
        recovery_firing_route_frames=@($rows|Where-Object {$_.arbitration.combat_intent.action -eq 'recover' -and $_.arbitration.move_source -like 'route*' -and ($_.sent_command.Buttons -band 1)}).Count
        scope='Current observed campaign threat interrupts exit travel; recovery/objective travel remains allowed'
    }
    if($report.campaign_engagement.engage_route_violations){throw 'Campaign travel ignored current combat intent'}
    $selected=@($rows|Where-Object {$_.tactic.source -eq 'live' -and $_.tactic.action -eq 'retreat'})
    $steps=@($selected|Where-Object {$_.arbitration.move_source -eq 'combat_retreat' -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0) -and ($_.sent_command.Buttons -band 1) -and $_.arbitration.aim_source -eq 'enemy'})
    $byFrame=@{};foreach($r in $rows){$byFrame[[int]$r.frame]=$r}
    if($ParasiteMixed){
        $detour=@($rows|Where-Object {$_.arbitration.move_source -eq 'combat_corner_detour' -and $_.arbitration.move_point})
        $moved=0
        foreach($row in $detour){$next=$byFrame[([int]$row.frame+1)];if(!$next){continue};if([math]::Sqrt([math]::Pow($next.self[0]-$row.self[0],2)+[math]::Pow($next.self[1]-$row.self[1],2)) -gt 2){$moved++}}
        $report.mixed_detour=@{frames=$detour.Count;actual_moving_steps=$moved;enemy_aim_attack_frames=@($detour|Where-Object {$_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)}).Count;scope='Bounded risk tradeoff past observed threats; no grenade splash protection or permanent group survival guarantee'}
        if($RequireMixedDetour -and ($detour.Count -lt 2 -or $moved -lt 2)){throw 'Actual native mixed corner detour absent'}
    }
    if($ParasiteWeapon -and $ParasiteLoadout -ne 'shotgun'){
        $expected=switch($ParasiteLoadout){stocked{'Machinegun'};scarce{'Machinegun'};hyper{'HyperBlaster'};rail{'Railgun'};default{'Blaster'}}
        $expectedModel=switch($ParasiteLoadout){stocked{'*/v_machn/*'};scarce{'*/v_machn/*'};hyper{'*/v_hyperb/*'};rail{'*/v_rail/*'};default{'Blaster'}}
        $request=@($rows|Where-Object {$_.weapon_request -eq "use $expected" -and $_.weapon_reason -eq 'parasite_retreat_range' -and $_.weapon -like '*/v_shotg/*' -and $_.ammo -gt 0 -and $_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.enemies|Where-Object {$_.class -eq 'monster_parasite' -and $_.clear_shot}).Count}|Select-Object -First 1)
        $fire=@($rows|Where-Object {$request.Count -and $_.frame -gt $request[0].frame -and ($_.sent_command.Buttons -band 1) -and $_.arbitration.aim_source -eq 'enemy' -and ($_.arbitration.move_source -eq 'combat_retreat' -or $_.arbitration.move_source -eq 'combat_corner_escape' -or $_.arbitration.move_source -eq 'combat_corner_detour' -or ($ParasiteHealthKit -and $_.goal -eq 'recover_health' -and $_.arbitration.move_source -like 'route*')) -and ($_.weapon -like $expectedModel -and ($expected -eq 'Blaster' -or $_.ammo -gt 0))})
        $movingAway=0
        foreach($row in $fire){
            $next=$byFrame[([int]$row.frame+1)];$target=@($row.enemies|Where-Object id -eq $row.arbitration.aim_entity)
            if(!$next -or $target.Count -ne 1){continue}
            $before=[math]::Sqrt([math]::Pow($row.self[0]-$target[0].origin[0],2)+[math]::Pow($row.self[1]-$target[0].origin[1],2))
            $after=[math]::Sqrt([math]::Pow($next.self[0]-$target[0].origin[0],2)+[math]::Pow($next.self[1]-$target[0].origin[1],2))
            if($after-$before -gt 2){$movingAway++}
        }
        $report.weapon_selection=@{expected=$expected;request_frames=@($request.frame);retreat_firing_frames=$fire.Count;actual_away_steps=$movingAway;minimum_health=($rows|Measure-Object health -Minimum).Minimum}
        $report.weapon_selection.shells_at_request=$(if($request.Count){[int](($request[0].inventory|Where-Object name -eq 'Shells'|Measure-Object count -Sum).Sum)}else{0})
        if($ParasiteLoadout -eq 'rail'){
            if(!$request.Count){throw 'Loaded Shotgun to Railgun request absent'}
            # Railgun may finish creating space before its equip animation ends.
            # Verify retreat -> settled ranged shot, not automatic-weapon fire while moving.
            $settling=@($rows|Where-Object {$_.frame -gt $request[0].frame -and $_.weapon -like $expectedModel -and $_.arbitration.limit_reason -eq 'rail_aim_settling' -and $_.arbitration.aim_source -eq 'enemy'})
            $railFire=@($rows|Where-Object {$_.frame -gt $request[0].frame -and $_.weapon -like $expectedModel -and $_.ammo -gt 0 -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)})
            $spaceSteps=0
            foreach($row in $rows){
                if(!$settling.Count -or $row.frame -ge $settling[0].frame -or $row.arbitration.move_source -ne 'combat_retreat' -or !$row.arbitration.combat_spacing.need_space -or $row.arbitration.aim_source -ne 'enemy'){continue}
                $next=$byFrame[([int]$row.frame+1)];$target=@($row.enemies|Where-Object id -eq $row.arbitration.aim_entity)
                if(!$next -or $target.Count -ne 1){continue}
                $before=[math]::Sqrt([math]::Pow($row.self[0]-$target[0].origin[0],2)+[math]::Pow($row.self[1]-$target[0].origin[1],2))
                $after=[math]::Sqrt([math]::Pow($next.self[0]-$target[0].origin[0],2)+[math]::Pow($next.self[1]-$target[0].origin[1],2))
                if($after-$before -gt 2){$spaceSteps++}
            }
            $unsafe=@($railFire|Where-Object {$_.arbitration.combat_spacing.distance -lt 320 -or $_.arbitration.aim_error_degrees -gt .5})
            $report.weapon_selection.rail_sequence=@{pre_aim_away_steps=$spaceSteps;settling_frames=$settling.Count;settled_firing_command_frames=$railFire.Count;minimum_firing_distance=($railFire.arbitration.combat_spacing|Measure-Object distance -Minimum).Minimum;unsafe_firing_commands=$unsafe.Count;scope='Create space before settled Railgun fire; no simultaneous moving-shot or long-range moving-target acceptance'}
            if($spaceSteps -lt 2 -or !$settling.Count -or $railFire.Count -lt 2 -or $unsafe.Count -or @($settling|Where-Object {$_.sent_command.Buttons -band 1}).Count -or $railFire[0].frame -le $settling[0].frame){throw 'Railgun retreat, aim suppression or settled safe-distance fire absent'}
        }elseif(!$request.Count -or $fire.Count -lt 2 -or (!$ParasiteHealthKit -and $movingAway -lt 2)){throw 'Loaded Shotgun switch and actual ranged firing movement absent'}
        if($ParasiteMixed){
            $report.weapon_selection.mixed_request_visible=@($request[0].enemies|Where-Object clear_shot).Count
            $report.weapon_selection.mixed_firing_frames=@($fire|Where-Object {@($_.enemies|Where-Object clear_shot).Count -ge 2}).Count
            if($report.weapon_selection.mixed_request_visible -lt 2 -or $report.weapon_selection.mixed_firing_frames -lt 2){throw 'Switch/fire in observed mixed group absent'}
        }
        if($ParasiteHealthKit){
            $approach=0;$heals=@()
            foreach($row in $rows){
                $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
                $kit=@($row.pickups|Where-Object {$_.class -eq 'item_health' -and [math]::Abs($_.origin[0]-32) -lt 1 -and [math]::Abs($_.origin[1]+424) -lt 1})
                if($kit.Count -eq 1 -and $next.health -gt $row.health -and [math]::Abs($next.self[0]-32) -lt 48 -and [math]::Abs($next.self[1]+424) -lt 48 -and @($next.pickups|Where-Object id -eq $kit[0].id).Count -eq 0){$heals+=@{frame=$next.frame;before=$row.health;after=$next.health;item=$kit[0].id}}
            }
            $recoveryFire=@($rows|Where-Object {$_.frame -ge $request[0].frame -and $_.goal -eq 'recover_health' -and $_.arbitration.move_source -like 'route*' -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)})
            foreach($row in $recoveryFire){
                $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
                $before=[math]::Sqrt([math]::Pow($row.self[0]-32,2)+[math]::Pow($row.self[1]+424,2));$after=[math]::Sqrt([math]::Pow($next.self[0]-32,2)+[math]::Pow($next.self[1]+424,2))
                if($before-$after -gt 2){$approach++}
            }
            $report.weapon_selection.request_health=$request[0].health
            $report.weapon_selection.recovery_firing_frames=$recoveryFire.Count
            $report.weapon_selection.recovery_approach_steps=$approach
            $report.weapon_selection.native_heal=$heals
            if($request[0].health -ge 45 -or $recoveryFire.Count -lt 2 -or $approach -lt 2 -or !$heals.Count -or $heals[0].frame -le $request[0].frame){throw 'Low-health switch with subsequent firing health recovery absent'}
        }
        if($ParasiteLoadout -eq 'scarce'){
            $hyperRequest=@($rows|Where-Object {$_.frame -gt $request[0].frame -and $_.weapon_request -eq 'use HyperBlaster' -and $_.weapon_reason -eq 'empty_weapon' -and $_.weapon -like '*/v_machn/*' -and $_.ammo -eq 0}|Select-Object -First 1)
            $blasterRequest=@($rows|Where-Object {$hyperRequest.Count -and $_.frame -gt $hyperRequest[0].frame -and $_.weapon_request -eq 'use Blaster' -and $_.weapon_reason -eq 'empty_weapon' -and $_.weapon -like '*/v_hyperb/*' -and $_.ammo -eq 0}|Select-Object -First 1)
            $unsafeSwitch=@($rows|Where-Object {$_.frame -gt $request[0].frame -and $_.weapon_request -eq 'use Shotgun' -and @($_.enemies|Where-Object {$_.class -eq 'monster_parasite' -and $_.clear_shot}).Count})
            $native=@($events|Where-Object {$_.attacker -eq $rows[0].self_entity -and ($_.target_class -eq $enemyClass -or ($ParasiteMixed -and $_.target_class -eq $ParasiteMixedClass)) -and $_.live_health_damage -gt 0})
            $mgDamage=[int](($native|Where-Object {$_.mod -eq 4 -and $_.frame -ge $request[0].frame}|Measure-Object live_health_damage -Sum).Sum)
            $hyperDamage=[int](($native|Where-Object {$hyperRequest.Count -and $_.mod -eq 10 -and $_.frame -ge $hyperRequest[0].frame}|Measure-Object live_health_damage -Sum).Sum)
            $blasterDamage=[int](($native|Where-Object {$blasterRequest.Count -and $_.mod -eq 1 -and $_.frame -ge $blasterRequest[0].frame}|Measure-Object live_health_damage -Sum).Sum)
            $report.weapon_selection.depletion=@{initial_bullets=[int](($request[0].inventory|Where-Object name -eq 'Bullets'|Measure-Object count -Sum).Sum);initial_cells=[int](($request[0].inventory|Where-Object name -eq 'Cells'|Measure-Object count -Sum).Sum);hyper_request_frame=$hyperRequest[0].frame;blaster_request_frame=$blasterRequest[0].frame;unsafe_shotgun_requests=$unsafeSwitch.Count;machinegun_health_damage=$mgDamage;hyper_health_damage=$hyperDamage;blaster_health_damage=$blasterDamage;scope='Authoritative empty ammo triggers ranged fallback in prepared mixed fight; not general resource optimization'}
            if(!$hyperRequest.Count -or !$blasterRequest.Count -or $unsafeSwitch.Count -or $mgDamage -le 0 -or $hyperDamage -le 0 -or $blasterDamage -le 0 -or $report.weapon_selection.depletion.initial_bullets -ne 5 -or $report.weapon_selection.depletion.initial_cells -ne 8){throw 'Native scarce-ammo ranged fallback chain absent'}
        }
        # Baseq2 local.h: MOD_BLASTER=1, MACHINEGUN=4, HYPERBLASTER=10, RAILGUN=11. This confirms
        # selected-weapon damage, not bullet accuracy or a specific move shot.
        $weaponMod=switch($ParasiteLoadout){stocked{4};scarce{4};hyper{10};rail{11};default{1}}
        $selectedDamage=@($events|Where-Object {$_.attacker -eq $rows[0].self_entity -and $_.target_class -eq 'monster_parasite' -and $_.mod -eq $weaponMod -and $_.frame -ge $request[0].frame})
        $report.weapon_selection.selected_weapon_health_damage=[int](($selectedDamage|Measure-Object live_health_damage -Sum).Sum)
        $report.weapon_selection.selected_weapon_kills=@($selectedDamage|Where-Object killed).Count
        if($ParasiteLoadout -ne 'scarce' -and ($report.weapon_selection.selected_weapon_health_damage -le 0 -or $report.weapon_selection.selected_weapon_kills -ne 1)){throw 'Native selected-weapon damage and kill absent'}
    }
    if($ParasiteWeapon -and $ParasiteLoadout -eq 'shotgun'){
        $fire=@($rows|Where-Object {$_.weapon -like '*/v_shotg/*' -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)})
        $hits=@($events|Where-Object {$_.attacker -eq $rows[0].self_entity -and $_.inflictor -eq $_.attacker -and $_.target_class -eq 'monster_parasite' -and $_.mod -eq 2 -and $_.live_health_damage -gt 0})
        $report.teacher_exercise=@{weapon='Shotgun';fixed_weapon=$true;aim_fire_commands=$fire.Count;native_health_damage=[int](($hits|Measure-Object live_health_damage -Sum).Sum);scope='Test-only fixed equipment, rules aim/fire; no learned weapon choice or general combat acceptance'}
        if(!$fire.Count -or !$hits.Count){throw 'Fixed Shotgun native aim/fire damage absent'}
    }
    if($CornerEscape){
        $escape=@($rows|Where-Object {$_.arbitration.move_source -eq 'combat_corner_escape' -and $_.arbitration.move_point -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)})
        $moved=0
        foreach($row in $escape){
            $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
            if([math]::Sqrt([math]::Pow($next.self[0]-$row.self[0],2)+[math]::Pow($next.self[1]-$row.self[1],2)) -gt 2){$moved++}
        }
        $report.corner_escape_frames=$escape.Count;$report.corner_escape_moving_steps=$moved
        $report.minimum_health=($rows|Measure-Object health -Minimum).Minimum
        if($escape.Count -lt 2 -or $moved -lt 2){throw 'Actual corner escape with preserved fire absent'}
    }
    $away=0;foreach($r in $steps){$next=$byFrame[([int]$r.frame+1)];$enemy=@($r.enemies|Where-Object id -eq $r.tactic.target);if(!$next -or $enemy.Count -ne 1){continue};$before=0.;$after=0.;foreach($i in 0..1){$before+=($r.self[$i]-$enemy[0].origin[$i])*($r.self[$i]-$enemy[0].origin[$i]);$after+=($next.self[$i]-$enemy[0].origin[$i])*($next.self[$i]-$enemy[0].origin[$i])};if([math]::Sqrt($after)-[math]::Sqrt($before) -gt 2){$away++}}
    $report.live_retreat_decisions=@($selected.tactic|Sort-Object map,frame -Unique).Count;$report.retreat_firing_frames=$steps.Count;$report.observed_away_steps=$away
    if(!$ParasiteWeapon -and !$CornerEscape -and !$Recovery -and !$Cover -and !$Circle -and (!$selected.Count -or $steps.Count -lt 2 -or $away -lt 2)){throw 'Model-selected firing retreat not observed'}
    if($Recovery){
        $fired=@($rows|Where-Object {$_.goal -eq 'recover_health' -and $_.arbitration.combat_intent.action -eq 'recover' -and $_.arbitration.move_source -like 'route*' -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1) -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0)})
        $heal=@();$approach=0
        foreach($row in $rows){
            $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
            # Stock item pickup: server playerstate HP gain near the kit,
            # with its observed entity disappearing in the same frame pair.
            $kit=@($row.pickups|Where-Object {$_.class -eq 'item_health' -and [math]::Abs($_.origin[0]-32) -lt 1 -and [math]::Abs($_.origin[1]+352) -lt 1})
            if($kit.Count -eq 1 -and $next.health -gt $row.health -and [math]::Abs($next.self[0]-32) -lt 48 -and [math]::Abs($next.self[1]+352) -lt 48 -and @($next.pickups|Where-Object id -eq $kit[0].id).Count -eq 0){$heal+=@{frame=$next.frame;before=$row.health;after=$next.health;item=$kit[0].id}}
        }
        foreach($row in $fired){$next=$byFrame[([int]$row.frame+1)];if(!$next){continue};$before=[math]::Sqrt([math]::Pow($row.self[0]-32,2)+[math]::Pow($row.self[1]+352,2));$after=[math]::Sqrt([math]::Pow($next.self[0]-32,2)+[math]::Pow($next.self[1]+352,2));if($before-$after -gt 2){$approach++}}
        $resumed=@($rows|Where-Object {$heal.Count -gt 0 -and $_.frame -gt $heal[0].frame -and $_.goal -eq 'reach_level_exit' -and $_.arbitration.combat_intent.action -eq 'engage' -and $_.arbitration.aim_source -eq 'enemy' -and ($_.sent_command.Buttons -band 1)})
        $report.recovery_firing_frames=$fired.Count;$report.recovery_approach_steps=$approach;$report.native_playerstate_heal=$heal;$report.post_heal_combat_frames=$resumed.Count
        $report.recovery_incoming_health_damage=[int](($events|Where-Object {$_.target -eq $rows[0].self_entity -and $_.attacker_class -eq $enemyClass -and $_.frame -ge $fired[0].frame -and $heal.Count -gt 0 -and $_.frame -le $heal[0].frame}|Measure-Object live_health_damage -Sum).Sum)
        $report.natural_recovery_triggered=@($fired|Where-Object {$_.health -lt 45}).Count -gt 0
        if($fired.Count -lt 2 -or $approach -lt 2 -or !$heal.Count -or !$resumed.Count){throw 'Recovery movement/fire, native healing or return to combat absent'}
        if($RecoveryHealth -ge 45 -and (!$report.natural_recovery_triggered -or $report.recovery_incoming_health_damage -le 0)){throw 'Natural recovery under native monster damage absent'}
    }
    if($GroupRetreat){
        $groupSteps=@($steps|Where-Object {@($_.enemies|Where-Object clear_shot).Count -ge 2})
        $observed=0;$safeAway=0;$unsafe=0
        foreach($row in $groupSteps){
            $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
            $observed++;$primaryAway=$false;$stepUnsafe=$false
            foreach($other in $row.enemies){
                $before=0.;$after=0.
                foreach($i in 0..1){$before+=($row.self[$i]-$other.origin[$i])*($row.self[$i]-$other.origin[$i]);$after+=($next.self[$i]-$other.origin[$i])*($next.self[$i]-$other.origin[$i])}
                $delta=[math]::Sqrt($after)-[math]::Sqrt($before)
                if($delta -lt -4){$unsafe++;$stepUnsafe=$true}
                if($other.id -eq $row.arbitration.combat_spacing.target -and $delta -gt 2){$primaryAway=$true}
            }
            if($primaryAway -and !$stepUnsafe){$safeAway++}
        }
        $report.group_retreat_frames=$groupSteps.Count;$report.group_observed_steps=$observed;$report.group_safe_away_steps=$safeAway;$report.group_unsafe_steps=$unsafe
        if($groupSteps.Count -lt 2 -or $safeAway -lt 2 -or $unsafe){throw 'Safe firing retreat from observed group absent'}
    }
    if($Circle){
        $chosen=@($rows|Where-Object {$_.tactic.source -eq 'live' -and $_.tactic.action -eq 'circle'})
        $fired=@($chosen|Where-Object {$_.arbitration.move_source -eq 'combat_circle' -and ($_.sent_command.Buttons -band 1) -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0) -and $_.arbitration.aim_source -eq 'enemy'})
        $lateral=0
        foreach($row in $fired){
            $next=$byFrame[([int]$row.frame+1)];$enemy=@($row.enemies|Where-Object id -eq $row.tactic.target)
            if(!$next -or $enemy.Count -ne 1){continue}
            $rx=$row.self[0]-$enemy[0].origin[0];$ry=$row.self[1]-$enemy[0].origin[1];$radius=[math]::Sqrt($rx*$rx+$ry*$ry)
            $dx=$next.self[0]-$row.self[0];$dy=$next.self[1]-$row.self[1]
            $after=[math]::Sqrt(($rx+$dx)*($rx+$dx)+($ry+$dy)*($ry+$dy))
            if($radius -gt 1 -and [math]::Abs($dx*$ry-$dy*$rx)/$radius -gt 2 -and [math]::Abs($after-$radius) -lt 8){$lateral++}
        }
        $report.model_circle_decisions=@($chosen.tactic|Sort-Object map,frame -Unique).Count;$report.circle_firing_frames=$fired.Count;$report.observed_lateral_steps=$lateral
        if(!$chosen.Count -or $fired.Count -lt 2 -or $lateral -lt 2){throw 'Model-selected lateral firing arc not observed'}
        $report.scope='Prepared single infantry fight with model-selected short lateral arcs; no full orbit, group or general campaign acceptance'
        if($Group){
            $groupFrames=@($fired|Where-Object {$_.enemies.Count -ge 2})
            $unsafe=0;$observedGroupSteps=0
            foreach($row in $groupFrames){
                $next=$byFrame[([int]$row.frame+1)];if(!$next){continue}
                $observedGroupSteps++
                foreach($other in $row.enemies|Where-Object {$_.id -ne $row.arbitration.aim_entity}){
                    $before=0.;$after=0.
                    foreach($i in 0..1){$before+=($row.self[$i]-$other.origin[$i])*($row.self[$i]-$other.origin[$i]);$after+=($next.self[$i]-$other.origin[$i])*($next.self[$i]-$other.origin[$i])}
                    if([math]::Sqrt($after)-[math]::Sqrt($before) -lt -4){$unsafe++}
                }
            }
            $report.group_circle_frames=$groupFrames.Count;$report.group_observed_steps=$observedGroupSteps;$report.group_unsafe_steps=$unsafe
            if($groupFrames.Count -lt 2 -or $observedGroupSteps -lt 2 -or $unsafe){throw 'Safe observed group circle steps absent'}
            $report.scope='Prepared two-infantry fight from 100HP; live model circle with observed flank separation, not general group tactics or campaign acceptance'
        }
    }
    if($Cover){
        $chosen=@($rows|Where-Object {$_.tactic.source -eq 'live' -and $_.tactic.action -eq 'cover'})
        $hidden=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_wait' -and @($_.enemies|Where-Object clear_shot).Count -eq 0})
        $fired=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_fire' -and ($_.sent_command.Buttons -band 1) -and $_.arbitration.aim_source -eq 'enemy'})
        $returned=@($rows|Where-Object {$_.arbitration.skill -eq 'combat_cover_return' -and $_.frame -gt $fired[0].frame -and @($_.enemies|Where-Object clear_shot).Count -eq 0})
        $report.model_cover_decisions=@($chosen.tactic|Sort-Object map,frame -Unique).Count;$report.cover_wait_frames=$hidden.Count;$report.cover_fire_frames=$fired.Count;$report.cover_return_frames=$returned.Count
        $report.cover_exposures=@($fired|Where-Object {$byFrame[([int]$_.frame-1)].arbitration.skill -ne 'combat_cover_fire'}).Count
        if(!$chosen.Count -or !$hidden.Count -or !$fired.Count -or !$returned.Count){throw 'Model-selected hide/fire/return cycle absent'}
        $report.scope=if($CoverFight){'Prepared single infantry fight: repeated model-selected cover exposures, attributed projectile damage and native kill; no group or general campaign acceptance'}else{'Prepared single infantry cover cycle; no group or general campaign acceptance'}
    }
    if($rows|Where-Object {$_.map -ne 'base1' -or $_.teammate -or $_.health -le 0}){throw 'Unexpected map/teammate/death'}
    . "$PSScriptRoot/read_damage_events.ps1"
    $events=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log'));$actor=$rows[0].self_entity
    $report.damage_summary=Measure-BotDamage $events $actor
    $report.final_health=$rows[-1].health;$report.trace=$trace
    if($Cover -or $Circle -or $GroupRetreat -or $Recovery){
        # Link the native projectile's spawn to the actual cover command, then
        # use its shot ID to attribute damage; do not count earlier attacks.
        . "$PSScriptRoot/read_projectile_ledger.ps1"
        $firingWindow=if($GroupRetreat){$groupSteps}else{$fired}
        $coverShots=@(Read-ProjectileLedger (Join-Path $OutputRoot 'server.log') $events|Where-Object {
            $shot=$_
            $shot.attacker -eq $actor -and @($firingWindow|Where-Object {$_.spawncount -eq $shot.spawncount -and $_.frame -eq $shot.spawn_frame}).Count -gt 0
        })
        if($Recovery){$report.recovery_projectiles=Measure-ProjectileLedger $coverShots $actor}elseif($GroupRetreat){$report.group_retreat_projectiles=Measure-ProjectileLedger $coverShots $actor}elseif($Circle){$report.circle_projectiles=Measure-ProjectileLedger $coverShots $actor}else{$report.cover_projectiles=Measure-ProjectileLedger $coverShots $actor}
        $coverDamage=@($coverShots.damage|Where-Object {$_.target_class -eq $enemyClass -and $_.live_health_damage -gt 0})
        if($Recovery){$report.recovery_health_damage=[int](($coverDamage|Measure-Object live_health_damage -Sum).Sum)}elseif($GroupRetreat){$report.group_retreat_health_damage=[int](($coverDamage|Measure-Object live_health_damage -Sum).Sum)}elseif($Circle){$report.circle_health_damage=[int](($coverDamage|Measure-Object live_health_damage -Sum).Sum)}else{$report.cover_window_health_damage=[int](($coverDamage|Measure-Object live_health_damage -Sum).Sum);$report.cover_window_damage_frames=@($coverDamage.frame)}
    }
    $report.kills=@($events|Where-Object {$_.attacker -eq $actor -and ($_.target_class -eq $enemyClass -or ($ParasiteMixed -and $_.target_class -eq $ParasiteMixedClass)) -and $_.killed}).Count
    if($Recovery -or $CornerEscape -or $ParasiteWeapon){
        $killFrame=($events|Where-Object {$_.attacker -eq $actor -and ($_.target_class -eq $enemyClass -or ($ParasiteMixed -and $_.target_class -eq $ParasiteMixedClass)) -and $_.killed}|Measure-Object frame -Maximum).Maximum
        $report.post_combat_route_frames=@($rows|Where-Object {$null -ne $killFrame -and $_.frame -gt $killFrame -and $_.goal -eq 'reach_level_exit' -and $_.arbitration.move_source -like 'route*' -and ($_.sent_command.Forward -ne 0 -or $_.sent_command.Side -ne 0)}).Count
        if(!$report.post_combat_route_frames){throw 'Campaign movement after recovered combat absent'}
    }
    $expectedKills=if($Group -or $GroupRetreat -or $ParasiteMixed){2}else{1}
    if($report.kills -ne $expectedKills -and !$Cover -and !$GroupRetreat){throw 'Native target kill absent'}
    if($CoverFight -and ($report.kills -ne 1 -or $report.cover_exposures -lt 2)){throw 'Repeated cover exposures and native kill absent'}
    if(@(Get-Content (Join-Path $OutputRoot 'server.log')|Where-Object {$_ -eq "g_test_seed ready version=1 seed=$Seed"}).Count -ne 1){throw 'Seed acknowledgement absent'}
    $commands=Get-Content (Join-Path $OutputRoot 'bot.err')
    if(@($commands|Select-String 'client command: teleport ').Count -ne 1 -or @($commands|Select-String 'client command: spawnentity ').Count -ne 1 -or ($commands -match 'client command: (god|kill|map|gamemap) ') -or @($commands|Select-String 'client command: give '|Where-Object { (!$ParasiteWeapon -and !$CornerEscape -and !$Cover -and !$Circle -and !$GroupRetreat -and !$Recovery) -or $_.Line -notlike "*client command: give health $initialHealth" }).Count){throw 'Unexpected setup/gameplay command'}
    if($Recovery -and $report.recovery_health_damage -le 0){throw 'Native monster damage from recovery firing window absent'}
    if($GroupRetreat){
        if($report.group_retreat_health_damage -le 0){throw 'Native damage from grouped retreat projectiles absent'}
        $report.group_retreat_maneuver_accepted=$true
        $report.group_fight_completed=$report.kills -eq 2
        if(!$report.group_fight_completed){throw 'Grouped retreat maneuver passed, but full native fight incomplete'}
    }
    if($Circle -and $report.circle_health_damage -le 0){throw 'Native damage from circle projectiles absent'}
    if($Cover -and $report.cover_window_health_damage -le 0){throw 'Native monster damage in cover firing window absent'};if($ParasiteWeapon){$report.scope="Prepared native Parasite and loaded Shotgun, automatic ranged-weapon selection from observed inventory, actual retreat/fire and native kill; no general group or campaign acceptance"}elseif($CornerEscape){$report.scope="Prepared native Parasite, skill $RecoverySkill at the recorded fatal corner, 65HP as after the used kit; Go bounded escape with actual movement/fire and native kill, no full recovery or general campaign acceptance"}elseif($Recovery){$report.scope="Prepared ${initialHealth}HP solo actor, skill $RecoverySkill vulnerable Parasite and native 25HP kit; Go recovery movement with attributed fire, item heal and resumed combat; no remembered hidden-item or general campaign acceptance"}elseif($GroupRetreat){$report.scope='Prepared two vulnerable infantry: close primary and rear flank, live retreat, actual spacing and attributed bolt damage; no mixed group or general campaign acceptance'}elseif(!$Cover -and !$Circle){$report.scope='Prepared single vulnerable native parasite, ordinary solo campaign commands; model-selected retreat plus shooting and observed movement, not general campaign acceptance'};$report.accepted=$true;$report.reason='accepted'
    if($ParasiteWeapon -and $ParasiteLoadout -eq 'shotgun'){$report.scope='Prepared fixed Shotgun teacher exercise; observed rules aim/fire and native hitscan damage, no learned policy or weapon-choice acceptance'}
}catch{
    $report.reason=$_.Exception.Message
    # Keep server ground truth even when gameplay acceptance fails early.
    if((Test-Path $trace) -and (Test-Path (Join-Path $OutputRoot 'server.log'))){
        try{
            $failureRows=@(Get-Content $trace|ForEach-Object {$_|ConvertFrom-Json})
            . "$PSScriptRoot/read_damage_events.ps1"
            $failureEvents=@(Read-DamageEvents (Join-Path $OutputRoot 'server.log'))
            $failureActor=$failureRows[0].self_entity
            $report.barrel_safety=Measure-BarrelSafety $failureRows $failureEvents
            $diagnostic=@{reason=$report.reason;minimum_health=($failureRows.health|Measure-Object -Minimum).Minimum;maps=@($failureRows.map|Sort-Object -Unique);last=($failureRows|Select-Object -Last 1 frame,map,self,health,goal,arbitration);damage=Measure-BotDamage $failureEvents $failureActor;kills=@($failureEvents|Where-Object {$_.attacker -eq $failureActor -and $_.target_class -eq $enemyClass -and $_.killed}).Count}
            $diagnosticPath=Join-Path $OutputRoot 'failure-diagnostic.json'
            $diagnostic|ConvertTo-Json -Depth 8|Set-Content $diagnosticPath
            $report.failure_diagnostic=$diagnosticPath
        }catch{$report.diagnostic_error=$_.Exception.Message}
    }
}
finally{
    foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}}
    if((Test-Path $trace) -and (Test-Path (Join-Path $OutputRoot 'server.log'))){
        try{$report.first_life_diagnostic=& "$PSScriptRoot/analyze_combat_first_life.ps1" -RunRoot $OutputRoot}
        catch{$report.first_life_diagnostic_error=$_.Exception.Message}
    }
    $report|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')
}
if(!$report.accepted){throw $report.reason}
