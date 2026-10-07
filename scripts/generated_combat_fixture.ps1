# Shared validation and native receipt for a sampled, pre-game fixture.
function Read-GeneratedCombatFixture([string]$Path,[int]$Seed,[string]$Loadout,[bool]$Mixed){
    $v=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json
    if($v.map -and $v.map -notin @('base1','base2')){throw 'Unsupported generated map'}
    if($v.version -ne 1 -or $v.engine_seed -ne $Seed -or $v.loadout -ne $Loadout -or $v.health -lt 1 -or $v.health -gt 100 -or $v.monsters.Count -ne (1+[int]$Mixed) -or $v.monsters[0].class -notin @('monster_parasite','monster_soldier','monster_infantry') -or ($Mixed -and ($v.monsters[0].class -ne 'monster_parasite' -or $v.monsters[1].class -ne 'monster_gunner'))){throw 'Invalid sampled combat fixture'}
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
    $expected=@(@{class='player';position=$Fixture.player;health=$Fixture.health})+@($Fixture.monsters|ForEach-Object {@{class=$_.class;position=$_.position;health=$(switch($_.class){monster_soldier{30};monster_infantry{100};monster_parasite{175};monster_gunner{175};default{throw 'Unknown stock monster health'}})}})
    $proof=@()
    foreach($e in $expected){
        $matching=@($entities|Where-Object {$_.class -eq $e.class -and $_.solid -eq 2 -and $_.health -eq $e.health -and [math]::Abs($_.position[0]-$e.position[0]) -le 1 -and [math]::Abs($_.position[1]-$e.position[1]) -le 1 -and [math]::Abs($_.position[2]-$e.position[2]) -le 1})
        if($matching.Count -ne 1 -or $matching[0].frame -ne 100){throw "Native generated start not confirmed: $($e.class)"}
        $proof+=$matching[0]
    }
    if(@($entities|Where-Object {$_.class -like 'monster_*'}).Count -ne $Fixture.monsters.Count){throw 'Unexpected native monster composition'}
    return @{confirmed=$true;entities=$proof;scope='Native frame-100 poses/resources/solid; BSP static hull preflight. Full engine equivalence and dynamic collision hull trace are not certified.'}
}
