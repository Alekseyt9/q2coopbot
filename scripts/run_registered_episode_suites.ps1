[CmdletBinding()]
param([object[]]$Episodes,[int[]]$Timescales,[int]$Port,[int]$Parallelism,[int]$Repetitions,[string]$OutputRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/check_player_personal_space.ps1"
. "$PSScriptRoot/check_low_health_pickup.ps1"
. "$PSScriptRoot/check_jail2_door.ps1"
. "$PSScriptRoot/check_jail3_descent.ps1"
. "$PSScriptRoot/check_bunk1_bridge.ps1"
. "$PSScriptRoot/check_coop_map_inventory.ps1"
$allowed=@{
    'jail3-live-stuck-20260930-204726'=@('jail3_descent','jail3_descent','scenario')
    'bunk1-dynamic-bridge-wrong-elevator'=@('bunk1_bridge','bunk1_bridge','scenario')
    'base2-tight-passage-player-block'=@('pickup','personal_space','scenario')
    'base2-live-missed-supershotgun-low-health'=@('pickup','low_health_pickup','scenario')
    'jail2-live-door-blocked-20260930'=@('jail2','jail2_door','scenario')
    'campaign-map-reconnect-inventory-loss'=@('inventory','coop_inventory','session')
}
foreach($episode in $Episodes) {
    $binding=$allowed[$episode.id]
    if(!$binding -or $episode.acceptance.fixture -ne $binding[0] -or $episode.acceptance.checker -ne $binding[1] -or $episode.acceptance.definition -ne $binding[2]) {throw "Invalid registered runner binding: $($episode.id)"}
}
$results=@()
foreach($group in $Episodes | Group-Object {$_.acceptance.fixture}) {
    $configDir=Join-Path $OutputRoot $group.Name
    New-Item -ItemType Directory -Path $configDir | Out-Null
    $lines=[collections.generic.list[string]]::new()
    $failure='';$suite=$null
    try {
        Write-Host "Registered fixture: $($group.Name), parallelism=$Parallelism, repetitions=$Repetitions"
        $runtime=if($group.Name -eq 'jail2') {& "$PSScriptRoot/prepare_jail2_door_runtime.ps1"} elseif($group.Name -eq 'jail3_descent') {& "$PSScriptRoot/prepare_jail3_descent_runtime.ps1"} elseif($group.Name -eq 'bunk1_bridge') {& "$PSScriptRoot/prepare_bunk1_bridge_runtime.ps1"} else {& "$PSScriptRoot/prepare_registered_pickup_runtime.ps1" -Profile $group.Name}
        $scenes=@();$sessions=@();$byName=@{}
        foreach($episode in $group.Group) {
            $path=Join-Path $repo $episode.scenario
            if($episode.acceptance.definition -eq 'session') {$sessions+=$path}
            else {$scenes+=$path;$def=Get-Content $path -Raw|ConvertFrom-Json;$byName[$def.name]=$episode}
        }
        $config=Join-Path $configDir 'suite.json'
        $scenes=@($scenes | ForEach-Object {[IO.Path]::GetRelativePath($configDir,$_ )})
        $sessions=@($sessions | ForEach-Object {[IO.Path]::GetRelativePath($configDir,$_ )})
        $runtimeRelative=[IO.Path]::GetRelativePath($configDir,[string]$runtime)
        @{version=1;scenarios=$scenes;sessions=$sessions;timescales=$Timescales;repetitions=$Repetitions;parallelism=$Parallelism;base_port=$Port;runtime_root=$runtimeRelative;tail_frames=5}|ConvertTo-Json -Depth 6|Set-Content $config -Encoding utf8
        try {
            & "$PSScriptRoot/run_scenario_suite.ps1" -Config $config | ForEach-Object {$lines.Add([string]$_);Write-Host $_}
        } catch {$failure=$_.Exception.Message}
        $line=@($lines | Where-Object {$_ -match '^Suite: (.+) \(\d+/\d+ passed\)$'} | Select-Object -Last 1)
        if(!$line.Count){throw $(if($failure){$failure}else{'Suite output missing'})}
        $null=$line[0] -match '^Suite: (.+) \(\d+/\d+ passed\)$'
        $suitePath=$Matches[1]
        $suite=Get-Content (Join-Path $suitePath 'suite-summary.json') -Raw|ConvertFrom-Json
        foreach($run in $suite.results) {
            $def=Get-Content (Join-Path $run.directory 'scenario.json') -Raw|ConvertFrom-Json
            $episode=if($run.kind -eq 'session'){$group.Group[0]}else{$byName[$def.name]}
            if(!$episode){throw 'Suite result has unknown scenario'}
            $item=[ordered]@{id=$episode.id;timescale=$run.timescale;repeat=$run.repeat;accepted=$false;reason='not_checked';artifacts=$run.directory;native_state=$run.state}
            try {
                if($failure -or !$suite.provenance_valid -or !$run.accepted){throw $(if($failure){$failure}else{'Native audit or scenario rejected'})}
                if($run.kind -eq 'session') {
                    $rows=@(Get-Content (Join-Path $run.directory 'observer.jsonl') | ForEach-Object {$_|ConvertFrom-Json})
                    $observerLog=(Get-Content (Join-Path $run.directory 'observer.log') -Raw) + "`n" + (Get-Content (Join-Path $run.directory 'observer.err') -Raw)
                    $item.metrics=Assert-CoopMapInventory $rows (Get-Content (Join-Path $run.directory 'server.log') -Raw) $observerLog
                } else {
                    $file=Get-ChildItem $run.directory -Filter '*-trace.jsonl' | Where-Object Name -NotLike '*human*' | Select-Object -First 1
                    if(!$file){throw 'Bot trace missing'}
                    $rows=@(Get-Content $file.FullName | ForEach-Object {$_|ConvertFrom-Json})
                    $item.metrics=switch($episode.acceptance.checker) {
                        'personal_space' {Assert-PlayerPersonalSpace $rows}
                        'low_health_pickup' {Assert-LowHealthPickup $rows}
                        'jail2_door' {Assert-Jail2DoorFollow $rows}
                        'jail3_descent' {Assert-Jail3Descent $rows}
                        'bunk1_bridge' {Assert-Bunk1BridgeFollow $rows}
                        default {throw 'Unknown registered checker'}
                    }
                }
                $item.accepted=$true;$item.reason='accepted'
            } catch {$item.reason=$_.Exception.Message}
            $results+=[pscustomobject]$item
        }
        if(@($suite.results).Count -ne $group.Count*$Timescales.Count*$Repetitions){throw 'Incomplete native result set'}
    } catch {
        $errorReason=$_.Exception.Message
        foreach($episode in $group.Group) {foreach($scale in $Timescales){foreach($repeat in 1..$Repetitions){
            $results+=[pscustomobject]@{id=$episode.id;timescale=$scale;repeat=$repeat;accepted=$false;reason=$errorReason;artifacts=$configDir}
        }}}
    }
    $results|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
    $Port+=$group.Count*$Timescales.Count*$Repetitions+10
}
