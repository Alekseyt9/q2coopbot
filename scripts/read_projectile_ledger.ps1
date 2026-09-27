# Player Blaster/HyperBlaster only. Entity slots may be reused; shot IDs may not
# be reused within one spawncount. Missing terminal records remain unresolved.
function Read-ProjectileLedger([string]$LogPath,$Damage) {
    $ready=$false;$shots=@{};$active=@{}
    foreach($line in Get-Content -LiteralPath $LogPath){
        if($line -eq 'g_test_projectile ready version=1'){$ready=$true;continue}
        if($line -notlike 'sv_test_projectile *' -and $line -notlike 'g_test_projectile *'){continue}
        if($line -notmatch '^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=(spawn|end) map=(\w+) frame=(\d+) shot=(\d+) entity=(\d+) (.*)$'){throw 'Malformed projectile event'}
        $m=$Matches;$generation=[int]$m[1];$frame=[int]$m[2];$kind=$m[3];$map=$m[4];$id=[uint32]$m[6];$entity=[int]$m[7];$tail=$m[8]
        if($id -eq 0){throw 'Zero projectile ID'}
        $key='{0}:{1}' -f $generation,$id
        $slot='{0}:{1}' -f $generation,$entity
        if($kind -eq 'spawn'){
            if($shots.ContainsKey($key)){throw 'Duplicate projectile ID'}
            if($active.ContainsKey($slot)){throw 'Projectile entity reused before its terminal event'}
            $active[$slot]=$id
            $number='(-?\d+(?:\.\d+)?)'
            if($tail -notmatch "^attacker=(\d+) mod=(1|10) x=$number y=$number z=$number vx=$number vy=$number vz=$number`$"){throw 'Malformed projectile spawn'}
            $v=$Matches
            $shots[$key]=[pscustomobject]@{spawncount=$generation;map=$map;shot=$id;entity=$entity;attacker=[int]$v[1];mod=[int]$v[2];spawn_frame=$frame
                origin=@([double]::Parse($v[3],[cultureinfo]::InvariantCulture),[double]::Parse($v[4],[cultureinfo]::InvariantCulture),[double]::Parse($v[5],[cultureinfo]::InvariantCulture))
                velocity=@([double]::Parse($v[6],[cultureinfo]::InvariantCulture),[double]::Parse($v[7],[cultureinfo]::InvariantCulture),[double]::Parse($v[8],[cultureinfo]::InvariantCulture))
                end_frame=$null;outcome='unresolved';target=0;damage=$null}
        }else{
            if(!$shots.ContainsKey($key)){throw 'Projectile end without spawn'}
            $s=$shots[$key]
            if($null -ne $s.end_frame -or $s.entity -ne $entity -or $s.map -ne $map -or $frame -lt $s.spawn_frame){throw 'Invalid projectile lifecycle'}
            if($tail -notmatch '^target=(\d+) outcome=(damage|geometry|sky|expired|freed)$'){throw 'Unknown projectile outcome'}
            $s.end_frame=$frame;$s.target=[int]$Matches[1];$s.outcome=$Matches[2]
            $active.Remove($slot)
        }
    }
    if(!$ready){throw 'Projectile telemetry not confirmed'}
    foreach($d in $Damage){
        if(!$d.shot){continue}
        $key='{0}:{1}' -f $d.spawncount,$d.shot
        if(!$shots.ContainsKey($key)){throw 'Damage has no matching projectile'}
        $s=$shots[$key]
        if($s.damage -or $s.entity -ne $d.inflictor -or $s.attacker -ne $d.attacker -or $s.mod -ne $d.mod -or $s.map -ne $d.map -or $d.frame -lt $s.spawn_frame -or ($null -ne $s.end_frame -and $d.frame -ne $s.end_frame)){throw 'Projectile damage attribution mismatch'}
        $s.damage=$d
    }
    foreach($s in $shots.Values){
        if($s.outcome -eq 'damage' -and (!$s.damage -or $s.target -ne $s.damage.target)){throw 'Projectile contact lacks matching damage event'}
        if($s.damage -and $s.outcome -notin @('damage','unresolved')){throw 'Damage conflicts with projectile outcome'}
    }
    $shots.Values | Sort-Object spawncount,shot
}

function Measure-ProjectileLedger($Shots,[int]$BotEntity){
    @($Shots|Where-Object attacker -eq $BotEntity|Group-Object mod|ForEach-Object {
        $s=@($_.Group)
        $hits=@($s|Where-Object {$_.damage.target_class -like 'monster_*' -and $_.damage.live_health_damage -gt 0})
        $unresolved=@($s|Where-Object outcome -eq 'unresolved').Count
        $unknown=@($s|Where-Object outcome -eq 'freed').Count
        [pscustomobject]@{mod=[int]$_.Name;projectiles_fired=$s.Count;live_monster_hit_projectiles=$hits.Count
            geometry=@($s|Where-Object outcome -eq 'geometry').Count;sky=@($s|Where-Object outcome -eq 'sky').Count
            expired=@($s|Where-Object outcome -eq 'expired').Count;unresolved=$unresolved;unknown_freed=$unknown
            corpse_contacts=@($s|Where-Object {$_.damage -and $_.damage.health_before -le 0}).Count
            zero_health_damage_contacts=@($s|Where-Object {$_.damage -and $_.damage.health_before -gt 0 -and $_.damage.live_health_damage -eq 0}).Count
            other_damage_contacts=@($s|Where-Object {$_.damage -and !($_.damage.target_class -like 'monster_*' -and $_.damage.live_health_damage -gt 0)}).Count
            live_monster_hit_fraction=$(if(!$unresolved -and !$unknown){$hits.Count/[double]$s.Count}else{$null})}
    })
}
