. "$PSScriptRoot/check_rail_aim.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
function Assert-RailDamage($Rows,$Events,[string]$Mode) {
    $aim=Assert-RailAim $Rows $Mode
    $rail=@($Rows | Where-Object {$_.weapon -like '*/v_rail/*' -and $_.health -gt 0 -and $_.map -eq 'base1'})
    $first=$rail[0]
    $bot=$first.self_entity
    if($bot -lt 1) {throw 'Observed bot identity missing'}
    $target=$first.arbitration.aim_entity
    $hits=@($Events | Where-Object {$_.spawncount -eq $first.spawncount -and $_.map -eq 'base1' -and $_.attacker -eq $bot -and $_.target -eq $target -and $_.target_class -eq 'monster_tank' -and $_.attacker_class -eq 'player' -and $_.mod -eq 11 -and $_.frame -ge $first.frame -and $_.frame -le $rail[-1].frame -and $_.live_health_damage -gt 0})
    if($Mode -eq 'rail_precision') {
        $early=@($Events | Where-Object {$_.spawncount -eq $first.spawncount -and $_.attacker -eq $bot -and $_.mod -eq 11 -and $_.frame -ge $Rows[0].frame -and $_.frame -lt $aim.fire_frame})
        if($early.Count) {throw 'Rail contact before precise fire permission'}
        if(!$hits.Count -or $hits[0].frame -lt $aim.fire_frame) {throw 'Observed rail shot did not produce attributed tank damage'}
        return [pscustomobject]@{aim=$aim;target=$target;damage_events=$hits.Count;target_health_damage=[int](($hits|Measure-Object live_health_damage -Sum).Sum);kills=@($hits|Where-Object killed).Count;scope='actual_target_damage_not_accuracy_comparison'}
    }
    $unsafe=@($Events | Where-Object {$_.spawncount -eq $first.spawncount -and $_.map -eq 'base1' -and $_.attacker -eq $bot -and $_.mod -eq 11 -and $_.frame -ge $first.frame -and $_.frame -lt $first.frame+10})
    if($unsafe.Count) {throw 'Rail damage contact during friendly guard'}
    [pscustomobject]@{aim=$aim;damage_contacts_in_guard=0;scope='first_ten_guard_frames'}
}
