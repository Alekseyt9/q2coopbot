[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config,[string]$PreparedOutput,[string]$ClientExe,[string]$ReporterExe)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$configPath=(Resolve-Path -LiteralPath $Config).Path
$base=Split-Path -Parent $configPath
$cfg=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
foreach($key in $cfg.PSObject.Properties.Name) {if($key -notin @('session','runtime_root','port','timescale','tail_frames')) {throw "Unknown session trial field: $key"}}
$tail=if($null -eq $cfg.tail_frames) {5} else {[int]$cfg.tail_frames}
if($tail -lt 2 -or $tail -gt 1000) {throw 'Session tail_frames must be 2..1000'}
if(($PreparedOutput -or $ClientExe -or $ReporterExe) -and -not ($PreparedOutput -and $ClientExe -and $ReporterExe)) {throw 'Prepared run requires output and both prebuilt executables'}
if($cfg.port -lt 1024 -or $cfg.port -gt 65534 -or $cfg.timescale -notin @(1,2)) {throw 'Invalid port/timescale'}
if(Get-NetUDPEndpoint -LocalPort $cfg.port -ErrorAction SilentlyContinue) {throw 'Session port is occupied'}
function Resolve-TrialPath([string]$Path) {
    if([IO.Path]::IsPathRooted($Path)) {return (Resolve-Path -LiteralPath $Path).Path}
    return (Resolve-Path -LiteralPath (Join-Path $base $Path)).Path
}
$source=Resolve-TrialPath $cfg.runtime_root
$sessionPath=Resolve-TrialPath $cfg.session
$definition=Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
$firstMap=$definition.phases[0].scenario.map
if($firstMap -notmatch '^[a-zA-Z0-9_]+$') {throw 'Invalid initial map'}
if($PreparedOutput) {
    $out=(Resolve-Path -LiteralPath $PreparedOutput).Path
    $runtime=$source
    $snapshot=$sessionPath
    $client=(Resolve-Path -LiteralPath $ClientExe).Path
    $reporter=(Resolve-Path -LiteralPath $ReporterExe).Path
} else {
$out=Join-Path $repo ('workspace/artifacts/session-trial-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$runtime=Join-Path $out 'runtime'
New-Item -ItemType Directory -Path (Join-Path $runtime 'baseq2/maps') -Force | Out-Null
Get-ChildItem -LiteralPath $source -File | Where-Object Extension -in @('.exe','.dll') | Copy-Item -Destination $runtime
Copy-Item -LiteralPath (Join-Path $source 'baseq2/game.dll') -Destination (Join-Path $runtime 'baseq2')
foreach($asset in Get-ChildItem -LiteralPath (Join-Path $source 'baseq2') -Recurse -File | Where-Object Extension -in @('.pak','.bsp','.aas')) {
    $relative=[IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName)
    $dest=Join-Path (Join-Path $runtime 'baseq2') $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    try {New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop | Out-Null} catch {Copy-Item -LiteralPath $asset.FullName -Destination $dest}
}
$snapshot=Join-Path $out 'session.json'
Copy-Item -LiteralPath $sessionPath -Destination $snapshot
$client=Join-Path $out 'q2coopbot.exe'
$reporter=Join-Path $out 'q2scenario-report.exe'
Push-Location $repo
try {
    go build -o $client ./cmd/q2coopbot; if($LASTEXITCODE) {throw 'Client build failed'}
    go build -o $reporter ./cmd/q2scenario-report; if($LASTEXITCODE) {throw 'Reporter build failed'}
} finally {Pop-Location}
}
$completion=Join-Path $out 'completion.json'
if((Test-Path $completion) -or (Test-Path (Join-Path $out 'actor-config.json'))) {throw 'Session requires a fresh run output'}
$wall=120
foreach($role in @('actor','observer')) {
    $name=if($role -eq 'actor') {'TestHuman'} else {'GoCoopMate'}
    $test=@{session=$snapshot;session_role=$role;scenario_result=$completion;idle=($role -eq 'actor')}
    if($role -eq 'observer') {$test.scenario_tail_frames=$tail}
    @{server=@{host='127.0.0.1';port=$cfg.port};client=@{name=$name;game_dir=(Join-Path $runtime 'baseq2')};run=@{duration="${wall}s";frame_paced=$true};output=@{trace_jsonl=(Join-Path $out "$role.jsonl")};test=$test} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out "$role-config.json") -Encoding utf8
}
$server=$null; $actor=$null; $observer=$null
$oldRcon=$env:Q2COOPBOT_TEST_RCON
$executionTimer=[diagnostics.stopwatch]::StartNew()
try {
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="+set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $($cfg.port) +set timescale $($cfg.timescale) +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_test_unlimited_loopback 1 +set sv_test_trace_client GoCoopMate +set sv_test_start_client GoCoopMate +map $firstMap"
    $serverLog=Join-Path $out 'server.log'
    $server=Start-Process -FilePath (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -RedirectStandardOutput $serverLog -RedirectStandardError (Join-Path $out 'server.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(-not (Test-Path $serverLog) -or -not (Select-String -LiteralPath $serverLog -Pattern 'sv_test_start_client ready:' -Quiet)) {if($server.HasExited -or (Get-Date) -gt $deadline) {throw 'Server startup failed'}; Start-Sleep -Milliseconds 100}
    $endpoints=@(Get-NetUDPEndpoint -OwningProcess $server.Id)
    if(-not ($endpoints | Where-Object {$_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $cfg.port}) -or ($endpoints | Where-Object {$_.LocalAddress -notin @('127.0.0.1','::1')})) {throw 'Server is not loopback-only'}
    $actor=Start-Process -FilePath $client -ArgumentList "--config `"$(Join-Path $out 'actor-config.json')`"" -RedirectStandardOutput (Join-Path $out 'actor.log') -RedirectStandardError (Join-Path $out 'actor.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    while(-not (Select-String -LiteralPath $serverLog -Pattern 'TestHuman entered the game' -Quiet)) {if($actor.HasExited -or (Get-Date) -gt $deadline) {throw 'Actor startup failed'}; Start-Sleep -Milliseconds 100}
    $observer=Start-Process -FilePath $client -ArgumentList "--config `"$(Join-Path $out 'observer-config.json')`"" -RedirectStandardOutput (Join-Path $out 'observer.log') -RedirectStandardError (Join-Path $out 'observer.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds($wall+5)
    while(-not $observer.HasExited) {if((Get-Date) -gt $deadline -or $server.HasExited -or $actor.HasExited) {throw "Session watchdog expired or server/actor exited; inspect $out"}; Start-Sleep -Milliseconds 100}
    if($observer.ExitCode -ne 0 -or -not (Test-Path $completion)) {throw "Session failed; inspect $out"}
    $end=Get-Content -LiteralPath $completion -Raw | ConvertFrom-Json
    if($end.state -ne 'completed' -or -not (Select-String -LiteralPath (Join-Path $out 'observer.err') -Pattern 'scenario_stop_frame=' -Quiet)) {throw 'Session did not complete with observer acknowledgement'}
    if(-not $actor.HasExited) {Stop-Process -Id $actor.Id; $actor.WaitForExit()}
    $reportConfig=Join-Path $out 'report-config.json'
    @{session=$snapshot;actor_trace='actor.jsonl';bot_trace='observer.jsonl';server_log='server.log';output='report.json'} | ConvertTo-Json | Set-Content -LiteralPath $reportConfig -Encoding utf8
    & $reporter --config $reportConfig *> (Join-Path $out 'reporter.log')
    if($LASTEXITCODE) {throw "Session analysis failed; inspect $out"}
    $executionTimer.Stop()
    @{orchestration_wall_seconds=$executionTimer.Elapsed.TotalSeconds;tail_frames=$tail} | ConvertTo-Json | Set-Content (Join-Path $out 'session-timing.json') -Encoding utf8
    Write-Output "Session accepted: $out"
} finally {
    $env:Q2COOPBOT_TEST_RCON=$oldRcon
    foreach($process in @($observer,$actor,$server)) {if($process -and -not $process.HasExited) {Stop-Process -Id $process.Id -ErrorAction SilentlyContinue}}
}
