$ErrorActionPreference='Stop'
. "$PSScriptRoot/read_damage_events.ps1"
$path=[IO.Path]::GetTempFileName()
try {
    $ready='g_test_damage ready version=1'
    $hit='g_test_damage version=1 map=base1 frame=40 attacker=2 target=81 inflictor=2 mod=11 health_before=50 health_after=-100 take=150 armor=0 power=0 protection=0 target_class=monster_tank attacker_class=player'
    $hit='sv_test_damage spawncount=42 server_frame=47 '+$hit
    $corpse=$hit.Replace('health_before=50 health_after=-100','health_before=-100 health_after=-250')
    $protected=$hit.Replace('target=81','target=2').Replace('attacker=2','attacker=81').Replace('health_before=50 health_after=-100 take=150','health_before=100 health_after=100 take=0').Replace('protection=0','protection=150').Replace('target_class=monster_tank attacker_class=player','target_class=player attacker_class=monster_tank')
    @($ready,$hit,$corpse,$protected) | Set-Content $path
    $events=@(Read-DamageEvents $path)
    if($events[0].frame -ne 47 -or $events[0].game_frame -ne 40 -or $events[0].spawncount -ne 42) {throw 'Network/game clocks confused'}
    $summary=Measure-BotDamage $events 2
    if($events.Count -ne 3 -or $summary.by_mod[0].monster_health_damage -ne 50 -or $summary.by_mod[0].monster_kills -ne 1 -or $summary.by_mod[0].monster_damage_events -ne 1 -or $summary.received_health_damage -ne 0) {throw 'Damage attribution/overkill/corpse/protection accounting failed'}
    Set-Content $path $ready
    if(@(Read-DamageEvents $path).Count -ne 0) {throw 'Empty telemetry failed'}
    foreach($bad in @($hit,($ready+"`n"+$hit.Replace('health_after=-100','health_after=0')),($ready+"`n"+$hit.Replace('version=1','version=2')),($ready+"`n"+'g_test_damage truncated'))) {
        Set-Content $path $bad
        $rejected=$false
        try {$null=@(Read-DamageEvents $path)}catch{$rejected=$true}
        if(!$rejected) {throw 'Invalid damage log accepted'}
    }
    'PASS: damage attribution, overkill, corpse, protection, empty log and 4 rejection controls'
} finally {Remove-Item -LiteralPath $path}
