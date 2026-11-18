[CmdletBinding()]
param([Parameter(Mandatory)][string]$SourceRoot,[Parameter(Mandatory)][string]$OutputRoot,[Parameter(Mandatory)][string]$Client,[Parameter(Mandatory)][string]$CheckpointTool,[int]$Port=31172,[ValidateRange(2,500)][int]$Frames=30,[Nullable[int]]$SamplingStream,[ValidateSet('','fire_only_v1','root_fire_only_v1','root_joint_only_v1')][string]$SamplingScope='')
$ErrorActionPreference='Stop'
if($SamplingScope -and $null -eq $SamplingStream){throw 'Sampling scope requires an explicit stream'}
$SourceRoot=(Resolve-Path -LiteralPath $SourceRoot).Path
$Client=(Resolve-Path -LiteralPath $Client).Path
$CheckpointTool=(Resolve-Path -LiteralPath $CheckpointTool).Path
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh branch output required'}
if($Port -lt 1024 -or $Port -gt 65534 -or (Get-NetUDPEndpoint -LocalPort $Port -ErrorAction SilentlyContinue)){throw 'Private branch port unavailable'}
$sourceConfig=Get-Content -LiteralPath (Join-Path $SourceRoot 'bot-config.json') -Raw|ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $SourceRoot 'combat-checkpoint/manifest.json') -Raw|ConvertFrom-Json
if($manifest.barrier.boundary -ne 'combat_predecision_v1'){throw 'Expected pre-decision combat source package'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
$runtime=Join-Path $OutputRoot 'runtime'
. "$PSScriptRoot/prepare_runtime.ps1" -FunctionsOnly
foreach($record in $manifest.runtime){
    $from=Join-Path (Join-Path $SourceRoot 'runtime') $record.name
    $to=Join-Path $runtime $record.name
    New-Item -ItemType Directory -Path (Split-Path $to -Parent) -Force|Out-Null
    if([IO.Path]::GetExtension($record.name) -eq '.pak' -and [IO.Path]::GetPathRoot($from) -eq [IO.Path]::GetPathRoot($to)){
        # Immutable archives may be shared; each branch still verifies the
        # checkpoint's exact runtime hashes before loading the native world.
        Install-RuntimePak $from $to (Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/build/runtime-paks')
    }else{
        Copy-Item -LiteralPath $from -Destination $to
    }
}
$checkpointDir=Join-Path $OutputRoot 'checkpoint'
$sourceCheckpoint=Join-Path $SourceRoot 'combat-checkpoint'
New-Item -ItemType Directory -Path $checkpointDir|Out-Null
# Checkpoint packages are immutable: save rejects an existing package and load
# copies native saves into a separate mutable runtime slot before using them.
# Share package files, not the runtime's working save directory.
foreach($file in Get-ChildItem -LiteralPath $sourceCheckpoint -Recurse -File){
    $relative=$file.FullName.Substring($sourceCheckpoint.Length).TrimStart('\','/')
    $target=Join-Path $checkpointDir $relative
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force|Out-Null
    if([IO.Path]::GetPathRoot($file.FullName) -eq [IO.Path]::GetPathRoot($target)){
        New-Item -ItemType HardLink -Path $target -Target $file.FullName -ErrorAction Stop|Out-Null
    }else{
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}
# Carry the same test cvars but no source placement or client commands.
Copy-Item -LiteralPath (Join-Path $SourceRoot 'runtime/baseq2/learning-test.cfg') -Destination (Join-Path $runtime 'baseq2/learning-test.cfg')
$instance='fragment-'+[guid]::NewGuid().ToString('N')
Add-Content -LiteralPath (Join-Path $runtime 'baseq2/learning-test.cfg') -Value "set sv_harness_instance $instance" -Encoding ascii
$env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
$report=@{accepted=$false;reason='not_run';loaded_combat_continuation=$false;source_checkpoint=$checkpointDir}
$server=$null;$bot=$null
function Invoke-BranchOperation([string]$Action,$Release=$null){
    $operation=@{version=1;action=$Action;server="127.0.0.1:$Port";instance=$instance;runtime_root=$runtime;checkpoint_dir=$checkpointDir;timeout_ms=15000}
    if($Action -eq 'load'){$operation.bind_participants=$true;$operation.hold_after_load=$true}
    if($Release){$operation.release=$Release}
    $path=Join-Path $OutputRoot "$Action-config.json"
    $operation|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $path -Encoding utf8NoBOM
    & $CheckpointTool --config $path 2> (Join-Path $OutputRoot "$Action.err") | Set-Content -LiteralPath (Join-Path $OutputRoot "$Action-result.json") -Encoding utf8NoBOM
    if($LASTEXITCODE){throw "Native branch $Action failed"}
    Get-Content -LiteralPath (Join-Path $OutputRoot "$Action-result.json") -Raw|ConvertFrom-Json
}
try{
    $seed=(Get-Content -LiteralPath (Join-Path $SourceRoot 'report.json') -Raw|ConvertFrom-Json).seed
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $Port +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set g_test_damage 1 +set g_test_seed $seed +exec learning-test.cfg +set sv_harness_instance $instance +map $($manifest.map)"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'server.log') -RedirectStandardError (Join-Path $OutputRoot 'server.err')
    # A read-only UDP challenge confirms this process is accepting requests.
    $probe=[Net.Sockets.UdpClient]::new()
    try{
        $probe.Connect('127.0.0.1',$Port);$probe.Client.ReceiveTimeout=150
        $packet=[byte[]](255,255,255,255)+[Text.Encoding]::ASCII.GetBytes("getchallenge`n")
        $deadline=[DateTime]::UtcNow.AddSeconds(15);$ready=$false
        do{
            if($server.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Branch server startup failed'}
            $null=$probe.Send($packet,$packet.Length)
            try{$remote=[Net.IPEndPoint]::new([Net.IPAddress]::Any,0);$reply=$probe.Receive([ref]$remote);$ready=[Text.Encoding]::ASCII.GetString($reply).Contains('challenge ')}catch [Net.Sockets.SocketException]{Start-Sleep -Milliseconds 50}
        }while(!$ready)
    }finally{$probe.Dispose()}
    $loaded=Invoke-BranchOperation 'load'
    if(!$loaded.rng_restored -or !$loaded.combat_restored -or $loaded.state -ne 'load_held'){throw 'Native phase/RNG/root restoration unproved'}
    $control=Join-Path $OutputRoot 'control.json'
    $botConfig=@{server=@{host='127.0.0.1';port=$Port};client=@{name=$sourceConfig.client.name;game_dir=(Join-Path $runtime 'baseq2')};models=@{system1=''};
        run=@{mode=$sourceConfig.run.mode;next_map=$sourceConfig.run.next_map;duration='120s';frame_paced=$true;game_frames=$Frames};combat=$sourceConfig.combat;
        output=@{trace_jsonl=(Join-Path $OutputRoot 'bot.jsonl');combat_capture=$true};
        test=@{synchronous=$true;combat_only=[bool]$sourceConfig.test.combat_only;combat_barrier=$true;weapon_switch_fixture=$sourceConfig.test.weapon_switch_fixture;checkpoint_restore=$checkpointDir;checkpoint_mode='combat';checkpoint_control=$control}}
    $config=Join-Path $OutputRoot 'bot-config.json'
    if($null -ne $SamplingStream){if($SamplingStream -lt 0){throw 'Sampling stream must be nonnegative'};$botConfig.test.checkpoint_sampling_stream=[long]$SamplingStream}
    if($SamplingScope){$botConfig.test.checkpoint_sampling_scope=$SamplingScope}
    $botConfig|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $config -Encoding utf8NoBOM
    $bot=Start-Process $Client -ArgumentList @('--config',('"'+$config+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputRoot 'bot.log') -RedirectStandardError (Join-Path $OutputRoot 'bot.err')
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    while(!(Test-Path -LiteralPath ($control+'.restored.json'))){
        if($bot.HasExited -or $server.HasExited -or [DateTime]::UtcNow -gt $deadline){throw 'Combat client restore receipt unavailable'}
        Start-Sleep -Milliseconds 20
    }
    $release=$loaded.load_anchor
    $release.controls=@($control)
    $null=Invoke-BranchOperation 'release' $release
    if(!$bot.WaitForExit(120000) -or $bot.ExitCode){throw 'Restored combat client failed'}
    $report.accepted=$true;$report.reason='combat_branch_collected';$report.native_rng_restored=$true;$report.native_phase_restored=$true
    $report.scope='Collected restored native combat continuation; equality must be checked against source observations and effects separately'
}catch{$report.reason=$_.Exception.Message;throw}finally{
    foreach($process in @($bot,$server)){if($process -and !$process.HasExited){Stop-Process -Id $process.Id;$null=$process.WaitForExit(5000)}}
    $report|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $OutputRoot 'report.json') -Encoding utf8NoBOM
}

