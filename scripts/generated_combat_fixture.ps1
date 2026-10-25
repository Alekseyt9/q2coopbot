# Shared validation and native receipt for a sampled, pre-game fixture.
function Read-GeneratedCombatFixture([string]$Path,[int]$Seed,[string]$Loadout,[bool]$Mixed){
    $v=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json
    if($v.map -and $v.map -notmatch '^[a-z][a-z0-9_]{0,31}$'){throw 'Invalid generated map name'}
    if($v.version -ne 1 -or $v.engine_seed -ne $Seed -or $v.loadout -ne $Loadout -or $v.health -lt 1 -or $v.health -gt 100 -or (!$Mixed -and $v.monsters.Count -ne 1) -or ($Mixed -and ($v.monsters.Count -lt 2 -or $v.monsters.Count -gt 4))){throw 'Invalid sampled combat fixture'}
    $classes=@($v.monsters.class)
    if($v.inventory){
        if($v.inventory.version -ne 1 -or $v.inventory.seed -ne $Seed -or $Loadout -ne 'weapons' -or $v.inventory.items.Count -ne 16 -or @($v.inventory.items.name|Sort-Object -Unique).Count -ne 16){throw 'Invalid generated inventory binding'}
        $names=@('Blaster','Shotgun','Super Shotgun','Machinegun','Chaingun','Grenade Launcher','Rocket Launcher','HyperBlaster','Railgun','BFG10K','Grenades','Bullets','Shells','Cells','Rockets','Slugs')
        foreach($item in $v.inventory.items){if($item.name -notin $names -or $item.count -lt 0 -or $item.count -gt 200){throw 'Invalid generated inventory item'}}
        if($v.inventory.initial_weapon -notin $names[0..10]){throw 'Invalid generated initial weapon'}
    }
    $admitted=@('monster_parasite','monster_soldier','monster_soldier_light','monster_soldier_ss','monster_infantry','monster_gunner','monster_berserk','monster_brain')
    if($v.monster_geometry){
        if($v.monster_geometry -ne 'stock_monster_hulls_habitats_v1'){throw 'Unknown monster geometry contract'}
        $admitted+=@('monster_chick','monster_medic','monster_mutant','monster_gladiator','monster_tank','monster_tank_commander','monster_supertank','monster_jorg','monster_makron','monster_flyer','monster_floater','monster_hover','monster_boss2','monster_flipper')
    }
    if(@($classes|Where-Object {$_ -notin $admitted}).Count){throw 'Unsupported sampled class'}
    foreach($position in @(@{p=$v.player})+@($v.monsters|ForEach-Object {@{p=$_.position}})){
        if($position.p.Count -ne 3){throw 'Invalid position dimension'}
        foreach($x in $position.p){$n=[double]$x;if([double]::IsNaN($n) -or [double]::IsInfinity($n) -or [math]::Abs($n) -gt 8192){throw 'Invalid sampled coordinate'}}
    }
    return $v
}
function Format-GeneratedPosition($Position,[string]$Separator=','){
    return (@($Position|ForEach-Object {([double]$_).ToString('0.###',[Globalization.CultureInfo]::InvariantCulture)}) -join $Separator)
}
function Confirm-GeneratedCombatStart($Fixture,[string]$ServerLog){
    $entities=@(Get-Content -LiteralPath $ServerLog|Select-String '^g_test_entity_start frame=(\d+) id=(\d+) inuse=1 class=(\S+) .*origin=([-\d.]+),([-\d.]+),([-\d.]+) .*solid=(\d+) health=(\d+)$'|ForEach-Object {
        $g=$_.Matches[0].Groups
        @{frame=[int]$g[1].Value;id=[int]$g[2].Value;class=$g[3].Value;position=@([double]::Parse($g[4].Value,[Globalization.CultureInfo]::InvariantCulture),[double]::Parse($g[5].Value,[Globalization.CultureInfo]::InvariantCulture),[double]::Parse($g[6].Value,[Globalization.CultureInfo]::InvariantCulture));solid=[int]$g[7].Value;health=[int]$g[8].Value}
    })
    $hp=@{monster_soldier_light=20;monster_soldier=30;monster_soldier_ss=40;monster_infantry=100;monster_parasite=175;monster_gunner=175;monster_berserk=240;monster_brain=300;monster_chick=175;monster_medic=300;monster_mutant=300;monster_gladiator=400;monster_tank=750;monster_tank_commander=1000;monster_supertank=1500;monster_jorg=3000;monster_makron=3000;monster_flyer=50;monster_floater=200;monster_hover=240;monster_boss2=2000;monster_flipper=50}
    $expected=@(@{class='player';position=$Fixture.player;health=$Fixture.health})+@($Fixture.monsters|ForEach-Object {if(!$hp.ContainsKey($_.class)){throw 'Unknown stock monster health'};@{class=$_.class;position=$_.position;health=$hp[$_.class]}})
    $proof=@()
    foreach($e in $expected){
        $matching=@($entities|Where-Object {$_.class -eq $e.class -and $_.solid -eq 2 -and $_.health -eq $e.health -and [math]::Abs($_.position[0]-$e.position[0]) -le 1 -and [math]::Abs($_.position[1]-$e.position[1]) -le 1 -and [math]::Abs($_.position[2]-$e.position[2]) -le 1})
        if($matching.Count -ne 1 -or $matching[0].frame -ne 100){throw "Native generated start not confirmed: $($e.class)"}
        $proof+=$matching[0]
    }
    if(@($proof.id|Sort-Object -Unique).Count -ne $expected.Count -or @($entities|Where-Object {$_.class -like 'monster_*'}).Count -ne $Fixture.monsters.Count){throw 'Unexpected native monster composition'}
    return @{confirmed=$true;entities=$proof;scope='Native frame-100 poses/resources/solid; BSP static hull preflight. Full engine equivalence and dynamic collision hull trace are not certified.'}
}
