[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$configPath=(Resolve-Path -LiteralPath $Config).Path
$base=Split-Path -Parent $configPath
$cfg=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
foreach($key in $cfg.PSObject.Properties.Name) {if($key -notin @('session','runtime_root','server_port','proxy_port','timescale','run_ms','after_ms','duration_ms','arm_phase','recovery_ms','follow_recovery_frames','require_upstream_loss')) {throw "Unknown network trial field: $key"}}
foreach($key in @('session','runtime_root','server_port','proxy_port','timescale','run_ms','after_ms','duration_ms','arm_phase','recovery_ms','follow_recovery_frames','require_upstream_loss')) {if($null -eq $cfg.$key) {throw "Missing network trial field: $key"}}
if($cfg.require_upstream_loss -isnot [bool]) {throw 'require_upstream_loss must be boolean'}
if($cfg.timescale -notin @(1,2) -or $cfg.server_port -lt 1024 -or $cfg.server_port -gt 65534 -or $cfg.proxy_port -lt 1024 -or $cfg.proxy_port -gt 65534 -or $cfg.server_port -eq $cfg.proxy_port) {throw 'Invalid network trial ports or timescale'}
if($cfg.run_ms -lt 3000 -or $cfg.run_ms -gt 120000 -or $cfg.after_ms -lt 0 -or $cfg.duration_ms -lt 1 -or $cfg.after_ms+$cfg.duration_ms+$cfg.recovery_ms -ge $cfg.run_ms -or $cfg.recovery_ms -lt 1 -or $cfg.follow_recovery_frames -lt 1 -or $cfg.follow_recovery_frames -gt 1000 -or $cfg.arm_phase -ne 0) {throw 'Invalid network trial timing; first phase only'}
foreach($port in @($cfg.server_port,$cfg.proxy_port)) {if(Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue) {throw "Port $port is occupied"}}
function Resolve-NetworkPath([string]$Path) {if([IO.Path]::IsPathRooted($Path)) {return (Resolve-Path -LiteralPath $Path).Path}; return (Resolve-Path -LiteralPath (Join-Path $base $Path)).Path}
$source=Resolve-NetworkPath $cfg.runtime_root
$sessionPath=Resolve-NetworkPath $cfg.session
$definition=Get-Content $sessionPath -Raw | ConvertFrom-Json
if(-not $definition.readiness_barrier -or $definition.phases[0].scenario.map -ne 'base1') {throw 'Network trial requires base1 first phase with readiness barrier'}
$out=Join-Path $repo ('workspace/artifacts/network-trial-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$runtime=Join-Path $out 'runtime'
New-Item -ItemType Directory -Path "$runtime/baseq2" -Force | Out-Null
Copy-Item -LiteralPath $configPath -Destination (Join-Path $out 'trial-config.json')
Copy-Item -LiteralPath $sessionPath -Destination (Join-Path $out 'session.json')
Get-ChildItem $source -File | Where-Object Extension -in @('.exe','.dll') | Copy-Item -Destination $runtime
Copy-Item -LiteralPath (Join-Path $source 'baseq2/game.dll') -Destination "$runtime/baseq2"
foreach($asset in Get-ChildItem (Join-Path $source 'baseq2') -Recurse -File | Where-Object Extension -in @('.pak','.bsp','.aas')) {
    $dest=Join-Path "$runtime/baseq2" ([IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName))
    New-Item -ItemType Directory (Split-Path -Parent $dest) -Force | Out-Null
    try {New-Item -ItemType HardLink -Path $dest -Target $asset.FullName | Out-Null} catch {Copy-Item -LiteralPath $asset.FullName -Destination $dest}
}
Push-Location $repo
try {foreach($name in @('q2coopbot','q2netfault','q2netfault-report')) {go build -o "$out/$name.exe" "./cmd/$name";if($LASTEXITCODE) {throw "Build failed: $name"}}} finally {Pop-Location}
$network=@{listen="127.0.0.1:$($cfg.proxy_port)";server="127.0.0.1:$($cfg.server_port)";after_ms=$cfg.after_ms;duration_ms=$cfg.duration_ms;decode_quake=$true;arm_barrier_dir='completion.json.barrier';arm_phase=0}
$relayConfig=$network.Clone(); $relayConfig.run_ms=$cfg.run_ms+1000; $relayConfig.events='network.jsonl'
$relayConfig | ConvertTo-Json | Set-Content "$out/relay.json" -Encoding utf8
foreach($role in @('actor','observer')) {
    $name=if($role -eq 'actor') {'TestHuman'} else {'GoCoopMate'}
    $port=if($role -eq 'actor') {$cfg.server_port} else {$cfg.proxy_port}
    $duration=if($role -eq 'actor') {$cfg.run_ms+5000} else {$cfg.run_ms}
    @{server=@{host='127.0.0.1';port=$port};client=@{name=$name;game_dir="$runtime/baseq2"};run=@{duration="${duration}ms";frame_paced=$true};output=@{trace_jsonl="$out/$role.jsonl"};test=@{idle=($role -eq 'actor');session="$out/session.json";session_role=$role;scenario_result="$out/completion.json";scenario_tail_frames=$(if($role -eq 'observer'){5}else{0})}} | ConvertTo-Json -Depth 6 | Set-Content "$out/$role-config.json" -Encoding utf8
}
@{network=$network;events='network.jsonl';bot_trace='observer.jsonl';actor_trace='actor.jsonl';server_log='server.log';output='report.json';recovery_ms=$cfg.recovery_ms;follow_recovery_frames=$cfg.follow_recovery_frames;require_upstream_loss=[bool]$cfg.require_upstream_loss} | ConvertTo-Json -Depth 6 | Set-Content "$out/report-config.json" -Encoding utf8
$server=$null;$relay=$null;$actor=$null;$observer=$null
$oldRcon=$env:Q2COOPBOT_TEST_RCON
$timer=[diagnostics.stopwatch]::StartNew()
try {
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="+set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set cheats 1 +set maxclients 4 +set port $($cfg.server_port) +set timescale $($cfg.timescale) +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set sv_test_start_client GoCoopMate +set sv_test_trace_client GoCoopMate +map base1"
    $server=Start-Process "$runtime/q2ded.exe" -ArgumentList $args -WorkingDirectory $runtime -WindowStyle Hidden -RedirectStandardOutput "$out/server.log" -RedirectStandardError "$out/server.err" -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(-not (Select-String -LiteralPath "$out/server.log" -Pattern 'sv_test_start_client ready:' -Quiet)) {if($server.HasExited -or (Get-Date) -gt $deadline) {throw 'Server startup failed'};Start-Sleep -Milliseconds 100}
    $endpoints=@(Get-NetUDPEndpoint -OwningProcess $server.Id)
    if(-not ($endpoints | Where-Object {$_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $cfg.server_port}) -or ($endpoints | Where-Object LocalAddress -notin @('127.0.0.1','::1'))) {throw 'Server is not loopback-only'}
    $actor=Start-Process "$out/q2coopbot.exe" -ArgumentList "--config `"$out/actor-config.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/actor.log" -RedirectStandardError "$out/actor.err" -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(-not (Select-String -LiteralPath "$out/server.log" -Pattern 'TestHuman entered the game' -Quiet)) {if($actor.HasExited -or (Get-Date) -gt $deadline) {throw 'Actor startup failed'};Start-Sleep -Milliseconds 100}
    $relay=Start-Process "$out/q2netfault.exe" -ArgumentList "--config `"$out/relay.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/relay.log" -RedirectStandardError "$out/relay.err" -PassThru
    $deadline=(Get-Date).AddSeconds(5)
    while(-not (Select-String -LiteralPath "$out/relay.log" -Pattern 'relay ready:' -Quiet)) {if($relay.HasExited -or (Get-Date) -gt $deadline) {throw 'Relay startup failed'};Start-Sleep -Milliseconds 50}
    $observer=Start-Process "$out/q2coopbot.exe" -ArgumentList "--config `"$out/observer-config.json`"" -WindowStyle Hidden -RedirectStandardOutput "$out/observer.log" -RedirectStandardError "$out/observer.err" -PassThru
    $deadline=(Get-Date).AddMilliseconds($cfg.run_ms+5000)
    while(-not $observer.HasExited) {if($server.HasExited -or $actor.HasExited -or $relay.HasExited -or (Get-Date) -gt $deadline) {throw 'Network trial process/watchdog failure'};Start-Sleep -Milliseconds 100}
    if($observer.ExitCode -ne 0) {throw 'Observer failed'}
    if(-not $relay.WaitForExit(3000) -or $relay.ExitCode -ne 0) {throw 'Relay failed'}
    if(-not $actor.HasExited) {Stop-Process -Id $actor.Id;$actor.WaitForExit()}
    & "$out/q2netfault-report.exe" --config "$out/report-config.json" *> "$out/reporter.log"
    $reportExit=$LASTEXITCODE
    $timer.Stop()
    @{orchestration_wall_seconds=$timer.Elapsed.TotalSeconds;scope='first_phase_timed_network_trial'} | ConvertTo-Json | Set-Content "$out/timing.json" -Encoding utf8
    Write-Output "Network trial: $out"
    if($reportExit) {throw 'Network report rejected; inspect report.json'}
} finally {$env:Q2COOPBOT_TEST_RCON=$oldRcon;foreach($process in @($observer,$actor,$relay,$server)){if($process -and -not $process.HasExited){Stop-Process -Id $process.Id -ErrorAction SilentlyContinue}}}
