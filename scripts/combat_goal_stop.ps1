# Supervisor-only native truth. This file never changes policy observations.
function Get-CombatGoalReceipt($Row, $Events, $Release, [string[]]$Classes, [string]$Map='base1') {
    if($Map -notin @('base1','base2') -or !$Row -or !$Release -or $Row.health -le 0 -or $Row.map -ne $Map -or $Row.connection -ne 1 -or $Row.combat_policy.observation.identity.life -ne 1){return $null}
    if($Row.spawncount -ne $Release.spawncount -or $Row.observation_frame -le $Release.frame){return $null}
    $live=@($Events | Where-Object {$_.spawncount -eq $Release.spawncount -and $_.map -eq $Row.map -and $_.frame -gt $Release.frame})
    if(@($live | Where-Object {$_.target -eq $Row.self_entity -and $_.target_class -eq 'player' -and $_.killed}).Count){return $null}
    $kills=@($live | Where-Object {$_.attacker -eq $Row.self_entity -and $_.attacker_class -eq 'player' -and $_.mod -ne 21 -and $_.killed -and $_.target_class -in $Classes})
    $selected=@(foreach($class in $Classes){
        $found=@($kills | Where-Object target_class -eq $class | Sort-Object frame | Select-Object -First 1)
        if($found.Count -ne 1){return $null};$found[0]
    })
    if(@($selected.target | Sort-Object -Unique).Count -ne $Classes.Count){return $null}
    $lastKill=[int](($selected | Measure-Object frame -Maximum).Maximum)
    # Keep a complete observed transition after the final kill, including its reward.
    if($Row.observation_frame -lt $lastKill+1){return $null}
    [pscustomobject]@{version='combat_goal_stop_v1';reason='combat_goal_complete';spawncount=$Row.spawncount;actor=$Row.self_entity;classes=$Classes;kill_frame=$lastKill;observed_frame=$Row.observation_frame;health=$Row.health;kills=@($selected | Select-Object frame,target,target_class)}
}

function Wait-CombatGoalOrExit($Bot, [string]$Root, [int]$TimeoutMilliseconds, [string[]]$Classes, [string]$Map='base1', [bool]$FirstDeath=$false) {
    . "$PSScriptRoot/read_damage_events.ps1"
    if($FirstDeath){. "$PSScriptRoot/combat_death_stop.ps1"}
    $clock=[Diagnostics.Stopwatch]::StartNew();$receipt=$null
    while(!$Bot.WaitForExit(50)){
        if($clock.ElapsedMilliseconds -gt $TimeoutMilliseconds){throw 'Bot timeout'}
        if($receipt){continue}
        $log=Join-Path $Root 'server.log';$trace=Join-Path $Root 'bot.jsonl'
        if(!(Test-Path $trace)){continue}
        $release=@(Get-Content -LiteralPath $log | Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
        if($release.Count -ne 1){continue}
        $row=$null
        try{$row=Get-Content -LiteralPath $trace -Tail 1 | ConvertFrom-Json -ErrorAction Stop}catch{continue} # Writer may still be appending a line.
        $context=@{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value}
        $events=@(Read-DamageEvents $log)
        $receipt=Get-CombatGoalReceipt $row $events $context $Classes $Map
        $receiptName='goal-stop.json';$stopReason='combat_goal_complete'
        if(!$receipt -and $FirstDeath){
            $receipt=Get-CombatFirstLifeDeathReceipt $row $events $context $Map
            $receiptName='death-stop.json';$stopReason='combat_first_life_death'
        }
        if($receipt){
            $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Root $receiptName) -Encoding utf8NoBOM
            Set-Content -LiteralPath (Join-Path $Root 'goal.stop') -Value $stopReason -Encoding ascii
        }
    }
}
