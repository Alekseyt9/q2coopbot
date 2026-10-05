[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunRoot)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/read_damage_events.ps1"
$root=(Resolve-Path -LiteralPath $RunRoot).Path
$rows=@(Get-Content -LiteralPath (Join-Path $root 'bot.jsonl') | ForEach-Object {$_|ConvertFrom-Json})
if(!$rows.Count -or $rows[0].self_entity -lt 1){throw 'Observed bot identity required'}
$start=$rows[0];$actor=[int]$start.self_entity
$life=@();$endReason='trace_end'
foreach($row in $rows){
    if($row.map -ne $start.map -or $row.spawncount -ne $start.spawncount -or $row.self_entity -ne $actor){$endReason='world_or_actor_changed';break}
    $life+=$row
    if($row.health -le 0){$endReason='first_observed_death';break}
}
$end=$life[-1]
$events=@(Read-DamageEvents (Join-Path $root 'server.log') | Where-Object {$_.map -eq $start.map -and $_.spawncount -eq $start.spawncount -and $_.frame -ge $start.frame -and $_.frame -le $end.frame})
$incoming=@($events|Where-Object target -eq $actor)
$aims=@(foreach($row in $life){
    if(!$row.arbitration.aim_entity){continue}
    $target=@($row.enemies|Where-Object {$_.id -eq $row.arbitration.aim_entity})
    if($target.Count -ne 1){continue}
    [pscustomobject]@{frame=$row.frame;entity=$target[0].id;class=$target[0].class;attack=[bool]($row.sent_command.Buttons -band 1)}
})
$motions=@(foreach($group in @($life|Group-Object {$_.arbitration.move_source})){
    $steps=0
    for($i=0;$i -lt $life.Count-1;$i++){
        $a=$life[$i];$b=$life[$i+1]
        if($a.arbitration.move_source -ne $group.Name -or $b.frame -ne ($a.frame+1)){continue}
        $dx=$b.self[0]-$a.self[0];$dy=$b.self[1]-$a.self[1]
        if([math]::Sqrt($dx*$dx+$dy*$dy) -gt 2){$steps++}
    }
    @{source=$group.Name;command_frames=$group.Count;actual_next_frame_steps=$steps}
})
$context=@($life|Select-Object -Last 16|ForEach-Object {
    @{frame=$_.frame;health=$_.health;weapon=$_.weapon;ammo=$_.ammo;gun_frame=$_.gun_frame;self=$_.self
      weapon_request=$_.weapon_request;weapon_reason=$_.weapon_reason;goal=$_.goal;move_source=$_.arbitration.move_source
      move_limit=$_.arbitration.move_limit_reason;aim_source=$_.arbitration.aim_source;attack=[bool]($_.sent_command.Buttons -band 1)
      aim_entity=$_.arbitration.aim_entity
      grenades=@($_.projectiles|Where-Object {$_.class -in @('grenade','hand_grenade')})
      visible_enemies=@($_.enemies|Where-Object {$_.clear_shot -eq $true}|Select-Object id,class,origin)}
})
$report=[ordered]@{
    version=1;scope='Offline first observed life only; server damage is diagnostic ground truth, never a bot observation. No survival acceptance or shot accuracy inferred.'
    map=$start.map;spawncount=$start.spawncount;bot_entity=$actor;start_frame=$start.frame;end_frame=$end.frame;end_reason=$endReason
    minimum_health=($life|Measure-Object health -Minimum).Minimum
    damage=(Measure-BotDamage $events $actor)
    incoming_by_mod=@(foreach($group in @($incoming|Group-Object mod)){
        @{mod=[int]$group.Name;health_damage=[int](($group.Group|Measure-Object live_health_damage -Sum).Sum);contacts=$group.Count}
    })
    incoming_by_attacker_class=@(foreach($group in @($incoming|Group-Object attacker_class)){
        @{attacker_class=$group.Name;health_damage=[int](($group.Group|Measure-Object live_health_damage -Sum).Sum);contacts=$group.Count}
    })
    outgoing_by_target_class=@(foreach($group in @($events|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*'}|Group-Object target_class)){
        @{target_class=$group.Name;health_damage=[int](($group.Group|Measure-Object live_health_damage -Sum).Sum);contacts=$group.Count;kills=@($group.Group|Where-Object killed).Count}
    })
    aim_by_class=@(foreach($group in @($aims|Group-Object class)){
        @{class=$group.Name;aim_frames=$group.Count;attack_command_frames=@($group.Group|Where-Object attack).Count}
    })
    weapon_requests=@($life|Where-Object {$_.weapon_request}|Select-Object frame,weapon,ammo,gun_frame,weapon_request,weapon_reason)
    movement=$motions;last_frames=$context
}
$output=Join-Path $root 'combat-first-life.json'
$report|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding utf8
[pscustomobject]@{report=$output;end_frame=$end.frame;end_reason=$endReason;received_health_damage=$report.damage.received_health_damage;monster_kills=[int](($report.damage.by_mod|Measure-Object monster_kills -Sum).Sum)}
