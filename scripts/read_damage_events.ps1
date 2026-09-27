# Server-only ground truth; never merge this data into bot observations.
function Read-DamageEvents([string]$LogPath) {
    $ready=$false
    foreach($line in Get-Content -LiteralPath $LogPath) {
        if($line -eq 'g_test_damage ready version=1') {$ready=$true;continue}
        if($line -notlike 'g_test_damage *' -and $line -notlike 'sv_test_damage *') {continue}
        if($line -notmatch '^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) (g_test_damage .*)$') {throw 'Damage event lacks network frame context'}
        $spawncount=[int]$Matches[1];$serverFrame=[int]$Matches[2];$line=$Matches[3]
        $pattern='^g_test_damage version=1 map=(\w+) frame=(\d+) attacker=(\d+) target=(\d+) inflictor=(\d+) mod=(\d+) health_before=(-?\d+) health_after=(-?\d+) take=(\d+) armor=(\d+) power=(\d+) protection=(\d+) target_class=(\w+) attacker_class=(\w+)(?: shot=(\d+))?$'
        if($line -notmatch $pattern) {throw "Malformed or unsupported damage event: $line"}
        $m=$Matches
        $event=[pscustomobject]@{
            map=$m[1];frame=$serverFrame;spawncount=$spawncount;game_frame=[int]$m[2];attacker=[int]$m[3];target=[int]$m[4];inflictor=[int]$m[5]
            mod=[int]$m[6];health_before=[int]$m[7];health_after=[int]$m[8];take=[int]$m[9]
            armor=[int]$m[10];power=[int]$m[11];protection=[int]$m[12];target_class=$m[13];attacker_class=$m[14]
            live_health_damage=0;killed=$false;shot=[uint32]$m[15]
        }
        if($event.health_before-$event.take -ne $event.health_after) {throw 'Inconsistent damage health values'}
        $event.live_health_damage=[math]::Min([math]::Max(0,$event.health_before),$event.take)
        $event.killed=$event.health_before -gt 0 -and $event.health_after -le 0
        $event
    }
    if(!$ready) {throw 'Server damage telemetry was not confirmed'}
}

function Measure-BotDamage($Events,[int]$BotEntity) {
    if($BotEntity -lt 1) {throw 'Bot entity must be observed'}
    $outgoing=@($Events | Where-Object attacker -eq $BotEntity)
    $incoming=@($Events | Where-Object target -eq $BotEntity)
    $weapons=@(foreach($group in @($outgoing | Group-Object mod)) {
        $live=@($group.Group | Where-Object {$_.target_class -like 'monster_*' -and $_.live_health_damage -gt 0})
        [pscustomobject]@{mod=[int]$group.Name;damage_contacts=$group.Count;monster_damage_events=$live.Count
            monster_health_damage=[int](($live | Measure-Object live_health_damage -Sum).Sum)
            monster_kills=@($live | Where-Object killed).Count}
    })
    [pscustomobject]@{bot_entity=$BotEntity;by_mod=$weapons
        received_health_damage=[int](($incoming | Measure-Object live_health_damage -Sum).Sum)
        self_health_damage=[int](($outgoing | Where-Object target -eq $BotEntity | Measure-Object live_health_damage -Sum).Sum)
        teammate_health_damage=[int](($outgoing | Where-Object {$_.target_class -eq 'player' -and $_.target -ne $BotEntity} | Measure-Object live_health_damage -Sum).Sum)
        scope='damage_events_not_shot_accuracy'}
}
