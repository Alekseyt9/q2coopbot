$ErrorActionPreference='Stop'
. "$PSScriptRoot/read_damage_events.ps1"
. "$PSScriptRoot/read_projectile_ledger.ps1"
$path=[IO.Path]::GetTempFileName()
try{
    $ready=@('g_test_damage ready version=1','g_test_projectile ready version=1')
    $spawn='sv_test_projectile spawncount=42 server_frame=40 g_test_projectile version=1 event=spawn map=base1 frame=30 shot=1 entity=81 attacker=2 mod=10 x=0.000 y=0.000 z=22.000 vx=1000.000 vy=0.000 vz=0.000'
    $hit='sv_test_damage spawncount=42 server_frame=44 g_test_damage version=1 map=base1 frame=34 attacker=2 target=316 inflictor=81 mod=10 health_before=30 health_after=-10 take=40 armor=0 power=0 protection=0 target_class=monster_soldier attacker_class=player shot=1'
    $end='sv_test_projectile spawncount=42 server_frame=44 g_test_projectile version=1 event=end map=base1 frame=34 shot=1 entity=81 target=316 outcome=damage'
    $spawn2=$spawn.Replace('server_frame=40','server_frame=45').Replace('shot=1','shot=2')
    $end2=$end.Replace('server_frame=44','server_frame=65').Replace('shot=1','shot=2').Replace('target=316 outcome=damage','target=0 outcome=expired')
    $spawn3=$spawn.Replace('server_frame=40','server_frame=70').Replace('shot=1','shot=3')
    $valid=@($ready)+@($spawn,$hit,$end,$spawn2,$end2,$spawn3)
    $valid|Set-Content $path
    $shots=@(Read-ProjectileLedger $path @(Read-DamageEvents $path))
    $m=@(Measure-ProjectileLedger $shots 2)[0]
    if($shots.Count -ne 3 -or $m.projectiles_fired -ne 3 -or $m.live_monster_hit_projectiles -ne 1 -or $m.expired -ne 1 -or $m.unresolved -ne 1 -or $null -ne $m.live_monster_hit_fraction){throw 'Projectile accounting failed'}
    foreach($case in @('missing_ready','duplicate_spawn','duplicate_end','missing_spawn','overlap','wrong_attacker','wrong_mod','wrong_entity','wrong_generation','wrong_outcome','missing_damage')){
        $lines=@($valid)
        switch($case){
            'missing_ready' {$lines=@($lines|Where-Object {$_ -ne 'g_test_projectile ready version=1'})}
            'duplicate_spawn' {$lines+= $spawn}
            'duplicate_end' {$lines+= $end}
            'missing_spawn' {$lines=@($lines|Where-Object {$_ -ne $spawn})}
            'overlap' {$lines=@($ready)+@($spawn,$spawn2,$hit,$end,$end2)}
            'wrong_attacker' {$lines=$lines.Replace($hit,$hit.Replace('attacker=2','attacker=3'))}
            'wrong_mod' {$lines=$lines.Replace($hit,$hit.Replace('mod=10','mod=1'))}
            'wrong_entity' {$lines=$lines.Replace($hit,$hit.Replace('inflictor=81','inflictor=82'))}
            'wrong_generation' {$lines=$lines.Replace($hit,$hit.Replace('spawncount=42','spawncount=43'))}
            'wrong_outcome' {$lines=$lines.Replace($end,$end.Replace('outcome=damage','outcome=geometry'))}
            'missing_damage' {$lines=@($lines|Where-Object {$_ -ne $hit})}
        }
        $lines|Set-Content $path
        $rejected=$false
        try{$null=@(Read-ProjectileLedger $path @(Read-DamageEvents $path))}catch{$rejected=$true}
        if(!$rejected){throw "Invalid projectile ledger accepted: $case"}
    }
    $ready|Set-Content $path
    if(@(Read-ProjectileLedger $path @()).Count){throw 'Empty projectile ledger failed'}
    'PASS: slot reuse, damage attribution, expiry, unresolved/empty logs and 11 rejection controls'
}finally{Remove-Item -LiteralPath $path}
