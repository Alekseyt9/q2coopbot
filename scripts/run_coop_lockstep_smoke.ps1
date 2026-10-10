[CmdletBinding()]
param([Parameter(Mandatory)][string]$OutputRoot,[int]$Port=29880,[int]$Seed=186000,[ValidateSet(0,100)][int]$ReleaseGameFrame=0)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/harness_manifest.ps1"
$sourceFingerprint=Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)
$nativeRoot=Join-Path (Split-Path $repo -Parent) 'yquake2'
$nativeFingerprint=Get-HarnessFingerprint (Get-HarnessNativeSourceRecords $nativeRoot)
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh output required'}
if(Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue){throw 'Occupied port'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$runtime=& "$PSScriptRoot/prepare_elevator_cycle_runtime.ps1" -Map base1 -RuntimeRoot (Join-Path $OutputRoot 'runtime')
# Protocol diagnostic excludes combat hazards, retaining stock BSP/PMove.
$entityPath=Join-Path $runtime 'baseq2/maps/base1.ent'
$entities=[IO.File]::ReadAllText($entityPath)
$entities=[regex]::Replace($entities,'(?ms)\{[^{}]*"classname"\s+"misc_explobox"[^{}]*\}\s*','')
[IO.File]::WriteAllText($entityPath,$entities,[Text.Encoding]::ASCII)
$client=Join-Path $OutputRoot 'q2coopbot.exe'
Push-Location $repo
try{go build -o $client ./cmd/q2coopbot;if($LASTEXITCODE){throw 'Build failed'}}finally{Pop-Location}
$settings=@('set sv_harness_instance pair-smoke','set sv_test_lockstep_client PairLearner','set sv_test_lockstep_peer PairLeader','set sv_test_signon_hold_frame 10','set g_test_combat_barrier 1','set g_test_combat_clients 2')
$settings+="set g_test_combat_release_frame $ReleaseGameFrame"
$settings|Set-Content -LiteralPath (Join-Path $runtime 'baseq2/pair.cfg') -Encoding ascii
$processes=@();$server=$null
$proof=@{accepted=$false;scope='Two-client command/tick smoke without monsters/barrels; stock BSP/PMove. No PPO export, reset equivalence or quality claim';seed=$Seed;timescale=2;release_game_frame=$ReleaseGameFrame;source_fingerprint=$sourceFingerprint;native_source_fingerprint=$nativeFingerprint;server_sha256=(Get-FileHash (Join-Path $runtime 'q2ded.exe')).Hash.ToLowerInvariant();client_sha256=(Get-FileHash $client).Hash.ToLowerInvariant()}
try{
 $arguments="-portable +set ip 127.0.0.1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set sv_test_unlimited_loopback 1 +set g_test_seed $Seed +set g_test_damage 1 +exec pair.cfg +map base1"
 $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $arguments -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
 Start-Sleep -Milliseconds 800
 foreach($role in 0..1){
  $name=if($role -eq 0){'PairLearner'}else{'PairLeader'}
  $position=if($role -eq 0){'-48,32,24'}else{'128,32,24'}
  $config=Join-Path $OutputRoot "$name.json"
  @{server=@{host='127.0.0.1';port=$Port};client=@{name=$name;game_dir=(Join-Path $runtime 'baseq2');aas_dir=(Join-Path $runtime 'baseq2/maps')};run=@{mode='companion';duration='60s';game_frames=$(if($ReleaseGameFrame){150}else{50});frame_paced=$true};combat=@{mode='rules'};test=@{synchronous=$true;combat_barrier=$true;teleport_map='base1';teleport=$position;initial_health=100;weapon_switch_fixture='parasite_blaster'};output=@{trace_jsonl=(Join-Path $OutputRoot "$name.jsonl");combat_capture=$true}}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $config -Encoding utf8NoBOM
  $processes+=Start-Process $client -ArgumentList "--config `"$config`"" -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot "$name.log") -RedirectStandardError (Join-Path $OutputRoot "$name.err")
  if($role -eq 0){Start-Sleep -Milliseconds 600}
 }
 $deadline=[DateTime]::UtcNow.AddSeconds(65)
 while(@($processes|Where-Object {!$_.HasExited}).Count){
  if($server.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Pair smoke timeout/server exit'}
  Start-Sleep -Milliseconds 100
 }
 if(@($processes|Where-Object ExitCode -ne 0).Count){throw 'Pair client failed'}
 $lines=Get-Content -LiteralPath (Join-Path $OutputRoot 'server.log')
 if(@($lines|Select-String 'sv_test_pair_step rejected').Count){throw 'Rejected paired commands'}
 $ends=@($lines|Select-String 'sv_test_pair_step version=1 phase=end spawncount=(\d+) frame=(\d+) role=(\d+) seq=(\d+) actor=(\d+)')
 $commands=@($lines|Select-String 'sv_test_pair_cmd version=1 spawncount=(\d+) frame=(\d+) role=(\d+) seq=(\d+) actor=(\d+)')
 if($ends.Count -lt 40 -or $ends.Count%2){throw 'Insufficient/partial completed pairs'}
 for($i=0;$i -lt $ends.Count;$i+=2){
  $a=$ends[$i].Matches[0];$b=$ends[$i+1].Matches[0]
  if($a.Groups[2].Value -ne $b.Groups[2].Value -or $a.Groups[3].Value -ne '0' -or $b.Groups[3].Value -ne '1'){throw 'Invalid pair closure order/frame'}
  foreach($end in @($a,$b)){
   $matching=@($commands|Where-Object {$m=$_.Matches[0]; $m.Groups[1].Value -eq $end.Groups[1].Value -and [int]$m.Groups[2].Value+1 -eq [int]$end.Groups[2].Value -and $m.Groups[3].Value -eq $end.Groups[3].Value -and $m.Groups[4].Value -eq $end.Groups[4].Value -and $m.Groups[5].Value -eq $end.Groups[5].Value})
   if($matching.Count -ne 1){throw 'Applied command/next frame does not match closure'}
  }
 }
 $proof.completed_pairs=$ends.Count/2;$proof.applied_commands=$commands.Count
 $proof.barrier_released=@($lines|Select-String 'g_test_combat_start game_frame=\d+ ready=2').Count -eq 1
 if(!$proof.barrier_released){throw 'Both participants did not release combat barrier'}
 if($ReleaseGameFrame){
  $release=@($lines|Select-String "^sv_test_combat spawncount=\d+ server_frame=\d+ g_test_combat_start game_frame=$ReleaseGameFrame ready=2 seed=$Seed`$")
  $rng=@($lines|Select-String "^g_test_rng_start game_frame=$ReleaseGameFrame phase=post_frame seed=$Seed cursor_before=\d+ cursor_after=256`$")
  $weapons=@($lines|Select-String "^g_test_weapon_start game_frame=$ReleaseGameFrame actor=([12]) weapon=Blaster gunframe_before=\d+ gunframe_after=9`$")
  if($release.Count -ne 1 -or $rng.Count -ne 1 -or $weapons.Count -ne 2 -or @($weapons|ForEach-Object {$_.Matches[0].Groups[1].Value}|Sort-Object -Unique).Count -ne 2){throw 'Paired fixed release/RNG/weapon phase verification failed'}
  $proof.fixed_release_verified=$true;$proof.post_frame_seed_verified=$true
 }
 $proof.source_unchanged=(Get-HarnessFingerprint (Get-HarnessSourceRecords $repo)) -eq $sourceFingerprint
 $proof.native_source_unchanged=(Get-HarnessFingerprint (Get-HarnessNativeSourceRecords $nativeRoot)) -eq $nativeFingerprint
 if(!$proof.source_unchanged -or !$proof.native_source_unchanged){throw 'Capture sources changed'}
 $proof.accepted=$true
}catch{$proof.reason=$_.Exception.Message;throw}finally{
 foreach($p in $processes){if(!$p.HasExited){Stop-Process -Id $p.Id}}
 if($server -and !$server.HasExited){Stop-Process -Id $server.Id}
 $proof|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $OutputRoot 'report.json') -Encoding utf8NoBOM
}
$proof
