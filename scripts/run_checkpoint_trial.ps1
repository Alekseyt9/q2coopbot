[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config)
$ErrorActionPreference='Stop'
$configPath=(Resolve-Path $Config).Path
$base=Split-Path $configPath -Parent
$cfg=Get-Content $configPath -Raw|ConvertFrom-Json
foreach($key in $cfg.PSObject.Properties.Name){if($key -notin @('runtime_root','client_exe','checkpoint_exe','output_root','port','timescale')){throw "Unknown checkpoint trial field: $key"}}
if($cfg.timescale -ne 2 -or $cfg.port -lt 1024 -or $cfg.port -gt 65534){throw 'Checkpoint trial requires 2x and a valid port'}
if(Get-NetUDPEndpoint -LocalPort $cfg.port -ErrorAction SilentlyContinue){throw 'Checkpoint port occupied'}
function Trial-Path([string]$Path){if([IO.Path]::IsPathRooted($Path)){return $Path};Join-Path $base $Path}
$source=(Resolve-Path (Trial-Path $cfg.runtime_root)).Path
$client=(Resolve-Path (Trial-Path $cfg.client_exe)).Path
$tool=(Resolve-Path (Trial-Path $cfg.checkpoint_exe)).Path
$out=Trial-Path $cfg.output_root
if(Test-Path $out){throw 'Checkpoint trial requires fresh output'}
$runtime=Join-Path $out 'runtime'
New-Item -ItemType Directory -Path (Join-Path $runtime 'baseq2/maps') -Force|Out-Null
Get-ChildItem $source -File|Where-Object Extension -In '.exe','.dll'|Copy-Item -Destination $runtime
Copy-Item (Join-Path $source 'baseq2/game.dll') (Join-Path $runtime 'baseq2/game.dll')
foreach($asset in Get-ChildItem (Join-Path $source 'baseq2') -File -Recurse|Where-Object Extension -In '.pak','.bsp','.aas','.ent'){
    $rel=[IO.Path]::GetRelativePath((Join-Path $source 'baseq2'),$asset.FullName)
    $dest=Join-Path (Join-Path $runtime 'baseq2') $rel
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force|Out-Null
    try{New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop|Out-Null}catch{Copy-Item $asset.FullName $dest}
}
$instance=[guid]::NewGuid().ToString('N')
$trace=Join-Path $out 'bot.jsonl'
@{server=@{host='127.0.0.1';port=$cfg.port};client=@{name='CheckpointBot';game_dir=(Join-Path $runtime 'baseq2')};run=@{duration='75s';frame_paced=$true};output=@{trace_jsonl=$trace};test=@{teleport_map='base2';teleport='304,-1888,-81.875';initial_health=38;hold_position=$true}}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'bot-config.json') -Encoding utf8
function Last-Row {
    if(Test-Path $trace){
        $last=$null
        foreach($line in Get-Content $trace -Tail 5){try{$last=$line|ConvertFrom-Json}catch{}}
        return $last
    }
    return $null
}
function Wait-Row([scriptblock]$Condition){
    $deadline=(Get-Date).AddSeconds(20)
    do{$row=Last-Row;if($row -and (& $Condition $row)){return $row};if($server.HasExited -or $bot.HasExited){throw 'Checkpoint server/client exited'};Start-Sleep -Milliseconds 100}while((Get-Date) -lt $deadline)
    throw 'Checkpoint observation timeout'
}
function Checkpoint-Operation([string]$Action){
    $path=Join-Path $out "$Action-config.json"
    @{version=1;action=$Action;server="127.0.0.1:$($cfg.port)";instance=$instance;runtime_root=$runtime;checkpoint_dir=(Join-Path $out 'checkpoint');timeout_ms=10000}|ConvertTo-Json|Set-Content $path -Encoding utf8
    $result=& $tool --config $path 2> (Join-Path $out "$Action.err")
    if($LASTEXITCODE){throw "Checkpoint $Action failed; inspect $Action.err"}
    $result|Set-Content (Join-Path $out "$Action-result.json") -Encoding utf8
    return $result|ConvertFrom-Json
}
$server=$null;$bot=$null;$oldRcon=$env:Q2COOPBOT_TEST_RCON
$report=[ordered]@{accepted=$false;reason='not_run';timescale=2;port=$cfg.port}
try{
    $env:Q2COOPBOT_TEST_RCON=[guid]::NewGuid().ToString('N')
    $args="-portable +set ip 127.0.0.1 +set noipx 1 +set dedicated 1 +set coop 1 +set deathmatch 0 +set cheats 1 +set maxclients 4 +set port $($cfg.port) +set timescale 2 +set rcon_password $env:Q2COOPBOT_TEST_RCON +set sv_harness_instance $instance +set sv_test_unlimited_loopback 1 +set sv_test_trace_client CheckpointBot +map base2`$base1"
    $server=Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $args -WorkingDirectory $runtime -RedirectStandardOutput (Join-Path $out 'server.log') -RedirectStandardError (Join-Path $out 'server.err') -WindowStyle Hidden -PassThru
    $deadline=(Get-Date).AddSeconds(20)
    do{Start-Sleep -Milliseconds 100;$eps=@(Get-NetUDPEndpoint -OwningProcess $server.Id -ErrorAction SilentlyContinue);if($server.HasExited -or (Get-Date) -gt $deadline){throw 'Server startup failed'}}while(!($eps|Where-Object {$_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $cfg.port}))
    if($eps|Where-Object LocalAddress -NotIn '127.0.0.1','::1'){throw 'Server not loopback-only'}
    $bot=Start-Process $client -ArgumentList "--config `"$(Join-Path $out 'bot-config.json')`"" -RedirectStandardOutput (Join-Path $out 'bot.log') -RedirectStandardError (Join-Path $out 'bot.err') -WindowStyle Hidden -PassThru
    $before=Wait-Row {param($r) $r.map -eq 'base2' -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    $save=Checkpoint-Operation 'save'
    if($save.state -ne 'saved' -or $save.gameplay_verified){throw 'Invalid native save result'}
    # Deliberately replace the whole game, not just a teleport. No arbitrary
    # console action comes from the trial definition.
    $udp=[Net.Sockets.UdpClient]::new();try{
        $udp.Connect('127.0.0.1',[int]$cfg.port)
        $bytes=[byte[]]@(255,255,255,255)+[Text.Encoding]::ASCII.GetBytes("rcon $env:Q2COOPBOT_TEST_RCON map base1`n")
        $null=$udp.Send($bytes,$bytes.Length)
    }finally{$udp.Dispose()}
    $changed=Wait-Row {param($r) $r.map -eq 'base1' -and $r.spawncount -ne $before.spawncount -and $r.health -eq 100 -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and !($r.inventory|Where-Object name -EQ 'Super Shotgun')}
    $load=Checkpoint-Operation 'load'
    if($load.state -ne 'load_acknowledged' -or $load.gameplay_verified){throw 'Load acknowledgement mislabeled as gameplay proof'}
    $after=Wait-Row {param($r) $r.map -eq 'base2' -and $r.spawncount -ne $before.spawncount -and $r.spawncount -ne $changed.spawncount -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    # Loading the same map again must still create a new generation and fresh
    # inventory, rather than being mistaken for a duplicate old observation.
    $reload=Checkpoint-Operation 'load'
    $again=Wait-Row {param($r) $r.map -eq 'base2' -and $r.spawncount -ne $before.spawncount -and $r.spawncount -ne $changed.spawncount -and $r.spawncount -ne $after.spawncount -and $r.health -eq 38 -and $r.on_ground -and $r.inventory_known -and $r.inventory_age_frames -le 1 -and ($r.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -and ($r.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}
    $distance=0;for($i=0;$i -lt 3;$i++){$distance+=[math]::Pow($before.self[$i]-$after.self[$i],2)}
    if([math]::Sqrt($distance) -gt 1){throw 'Saved position not restored'}
    for($i=0;$i -lt 3;$i++){if([math]::Abs($before.self[$i]-$again.self[$i]) -gt 1){throw 'Same-map reload position not restored'}}
    $null=Wait-Row {param($r) $r.spawncount -eq $again.spawncount -and $r.frame -ge $again.frame+5 -and $r.health -eq 38}
    $stable=@(Get-Content $trace|ForEach-Object {try{$_|ConvertFrom-Json}catch{}}|Where-Object {$_.spawncount -eq $again.spawncount -and $_.inventory_known -and $_.inventory_age_frames -le 1})
    if($stable.Count -lt 2){throw 'Insufficient fresh restored inventory rows'}
    if(@($stable|Where-Object {$_.health -ne 38 -or !($_.inventory|Where-Object {$_.name -eq 'Super Shotgun' -and $_.count -eq 1}) -or !($_.inventory|Where-Object {$_.name -eq 'Shells' -and $_.count -eq 10})}).Count){throw 'Unstable restored health/inventory'}
    $log=Get-Content (Join-Path $out 'bot.err') -Raw
    if(@([regex]::Matches($log,'client command: teleport ')).Count -ne 1 -or @([regex]::Matches($log,'client command: give health ')).Count -ne 1){throw 'Test placement/health reapplied after load'}
    $report.accepted=$true;$report.reason='accepted';$report.before=$before;$report.changed=$changed;$report.after=$after;$report.same_map_reload=$again;$report.fresh_inventory_rows=$stable.Count;$report.position_error=[math]::Sqrt($distance);$report.server_pid=$server.Id;$report.client_pid=$bot.Id
}catch{$report.reason=$_.Exception.Message}finally{
    foreach($p in @($bot,$server)){if($p -and !$p.HasExited){Stop-Process -Id $p.Id -ErrorAction SilentlyContinue;$p.WaitForExit()}}
    $env:Q2COOPBOT_TEST_RCON=$oldRcon
    $report|ConvertTo-Json -Depth 30|Set-Content (Join-Path $out 'report.json') -Encoding utf8
}
if(!$report.accepted){throw "Checkpoint rejected: $($report.reason); inspect $out"}
Write-Output "Checkpoint accepted: $out"
