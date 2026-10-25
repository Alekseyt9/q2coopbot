# Supervisor-only native truth. This file never changes policy observations.
function Get-CombatGoalKillEvidence([int]$Actor, $Events, $Release, [string[]]$Classes, [string]$Map) {
    $kills=@($Events | Where-Object {$_.spawncount -eq $Release.spawncount -and $_.map -eq $Map -and $_.frame -ge $Release.frame -and
        $_.attacker -eq $Actor -and $_.attacker_class -eq 'player' -and $_.mod -ne 21 -and $_.killed -and $_.health_before -gt 0 -and $_.target_class -in $Classes})
    $selected=@(foreach($group in @($Classes|Group-Object)){
        $found=@($kills | Where-Object target_class -eq $group.Name | Sort-Object frame | Group-Object target | ForEach-Object {$_.Group[0]} | Sort-Object frame | Select-Object -First $group.Count)
        if($found.Count -ne $group.Count){return $null};$found
    })
    if(!$selected.Count -or @($selected.target | Sort-Object -Unique).Count -ne $Classes.Count){return $null}
    [pscustomobject]@{kills=$selected;frame=[int](($selected | Measure-Object frame -Maximum).Maximum)}
}

function Get-CombatGoalReceipt($Row, $Events, $Release, [string[]]$Classes, [string]$Map='base1', $Objectives=$null) {
    if($Map -notmatch '^[a-z][a-z0-9_]{0,31}$' -or !$Row -or !$Release -or $Row.health -le 0 -or $Row.map -ne $Map -or $Row.connection -ne 1 -or $Row.combat_policy.observation.identity.life -ne 1){return $null}
    if($Row.spawncount -ne $Release.spawncount -or $Row.observation_frame -le $Release.frame){return $null}
    # The first released ClientThink can kill inside the release server frame.
    # Its complete following observation is still required below.
    $live=@($Events | Where-Object {$_.spawncount -eq $Release.spawncount -and $_.map -eq $Row.map -and $_.frame -ge $Release.frame -and $_.frame -le $Row.observation_frame})
    if(@($live | Where-Object {$_.target -eq $Row.self_entity -and $_.target_class -eq 'player' -and $_.killed}).Count){return $null}
    $evidence=Get-CombatGoalKillEvidence $Row.self_entity $live $Release $Classes $Map
    if(!$evidence){return $null}
    $selected=$evidence.kills;$lastKill=$evidence.frame
    # Keep a complete observed transition after the final kill, including its reward.
    if($Row.observation_frame -lt $lastKill+1){return $null}
    $receipt=[pscustomobject]@{version='combat_goal_stop_v1';reason='combat_goal_complete';spawncount=$Row.spawncount;actor=$Row.self_entity;classes=$Classes;kill_frame=$lastKill;observed_frame=$Row.observation_frame;health=$Row.health;kills=@($selected | Select-Object frame,target,target_class)}
    if($Release.require_objective){
        $state=@($Objectives|Where-Object {$_.spawncount -eq $Row.spawncount -and $_.map -eq $Map -and $_.frame -eq $Row.observation_frame})
        if($state.Count -ne 1 -or $state[0].alive -ne 0 -or $state[0].pending -ne 0){return $null}
        $receipt.version='combat_goal_stop_v2'
        $receipt|Add-Member -NotePropertyName native_objective -NotePropertyValue $state[0]
    }
    $receipt
}

function Get-CombatEarliestGoalReceipt($Rows, $Events, $Release, [string[]]$Classes, [string]$Map='base1', $Objectives=$null) {
    # Rows arrive in trace order. Retain observations until native events catch
    # up; choosing the newest row makes the terminal depend on polling timing.
    $actorRow=$null
    foreach($candidate in $Rows){
        if($candidate.spawncount -eq $Release.spawncount -and $candidate.map -eq $Map -and $candidate.connection -eq 1 -and
            $candidate.observation_frame -gt $Release.frame -and $candidate.combat_policy.observation.identity.life -eq 1){$actorRow=$candidate;break}
    }
    if(!$actorRow){return $null}
    $evidence=Get-CombatGoalKillEvidence $actorRow.self_entity $Events $Release $Classes $Map
    if(!$evidence){return $null}
    foreach($candidate in $Rows){
        if($candidate.observation_frame -le $evidence.frame){continue}
        $receipt=Get-CombatGoalReceipt $candidate $Events $Release $Classes $Map $Objectives
        if($receipt){return $receipt}
    }
}

function Get-CombatVerifiedSavedGoalReceipt($Saved, $Rows, $Events, $Release, [string[]]$Classes, [string]$Map='base1', $Objectives=$null) {
    # Final process shutdown may append a later death. Verify the already
    # selected earliest complete transition, rather than the last trace row.
    if(!$Saved){return $null}
    $verified=Get-CombatEarliestGoalReceipt $Rows $Events $Release $Classes $Map $Objectives
    if(!$verified){return $null}
    foreach($key in @('version','reason','spawncount','actor','kill_frame','observed_frame','health')){
        if($Saved.$key -ne $verified.$key){return $null}
    }
    if(($Saved.classes -join '|') -ne ($verified.classes -join '|')){return $null}
    $savedKills=@($Saved.kills|Select-Object frame,target,target_class)|ConvertTo-Json -Depth 6 -Compress
    $verifiedKills=@($verified.kills|Select-Object frame,target,target_class)|ConvertTo-Json -Depth 6 -Compress
    if($savedKills -ne $verifiedKills){return $null}
    if($Release.require_objective -and ($Saved.native_objective|ConvertTo-Json -Compress) -ne ($verified.native_objective|ConvertTo-Json -Compress)){return $null}
    return $verified
}

function Wait-CombatGoalOrExit($Bot, [string]$Root, [int]$TimeoutMilliseconds, [string[]]$Classes, [string]$Map='base1', [bool]$FirstDeath=$false, [bool]$RequireObjective=$false) {
    . "$PSScriptRoot/read_damage_events.ps1"
    . "$PSScriptRoot/combat_trace_reader.ps1"
    if($RequireObjective){. "$PSScriptRoot/read_combat_objective.ps1"}
    if($FirstDeath){. "$PSScriptRoot/combat_death_stop.ps1"}
    $clock=[Diagnostics.Stopwatch]::StartNew();$receipt=$null;$row=$null;$pendingDeath=$null
    $trace=Join-Path $Root 'bot.jsonl';$log=Join-Path $Root 'server.log'
    $cursor=New-CombatTraceCursor $trace
    $observations=[Collections.Generic.List[object]]::new()
    while(!$Bot.WaitForExit(50)){
        if($clock.ElapsedMilliseconds -gt $TimeoutMilliseconds){throw 'Bot timeout'}
        if($receipt){continue}
        if(!(Test-Path $trace)){continue}
        $release=@(Get-Content -LiteralPath $log | Select-String '^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=\d+ ready=1 seed=\d+$')
        if($release.Count -ne 1){continue}
        $context=@{spawncount=[int]$release[0].Matches[0].Groups[1].Value;frame=[int]$release[0].Matches[0].Groups[2].Value}
        $context.require_objective=$RequireObjective
        foreach($nextRow in @(Read-CombatTraceRows $cursor 8388608)){
                # Keep only supervisor evidence, not full feature/action tensors.
                $observations.Add([pscustomobject]@{observation_frame=$nextRow.observation_frame;spawncount=$nextRow.spawncount;
                    self_entity=$nextRow.self_entity;health=$nextRow.health;map=$nextRow.map;connection=$nextRow.connection;
                    combat_policy=@{observation=@{identity=$nextRow.combat_policy.observation.identity;health=$nextRow.combat_policy.observation.health}}})
                $row=$nextRow
                if($FirstDeath -and !$pendingDeath -and $nextRow.combat_policy.observation -and $null -ne $nextRow.health -and $nextRow.health -le 0 -and
                    $nextRow.combat_policy.observation.identity.life -eq 1 -and $nextRow.observation_frame -gt $context.frame -and
                    $nextRow.spawncount -eq $context.spawncount -and $nextRow.connection -eq 1 -and $nextRow.map -eq $Map){
                    # Retain the observation even if native damage arrives later
                    # or more trace rows already show the respawned actor.
                    $pendingDeath=$nextRow
                }
        }
        if(!$row){continue}
        $events=@(Read-DamageEvents $log)
        $objectives=if($RequireObjective){@(Read-CombatObjective $log)}else{@()}
        $receipt=Get-CombatEarliestGoalReceipt $observations $events $context $Classes $Map $objectives
        $receiptName='goal-stop.json';$stopReason='combat_goal_complete'
        if(!$receipt -and $FirstDeath){
            $receipt=Get-CombatFirstLifeDeathReceipt $pendingDeath $events $context $Map
            $receiptName='death-stop.json';$stopReason='combat_first_life_death'
        }
        if($receipt){
            $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Root $receiptName) -Encoding utf8NoBOM
            Set-Content -LiteralPath (Join-Path $Root 'goal.stop') -Value $stopReason -Encoding ascii
        }
    }
}
