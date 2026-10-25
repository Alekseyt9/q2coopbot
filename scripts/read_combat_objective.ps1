# End-of-frame native supervisor truth; never a policy observation.
function Read-CombatObjective([string]$Path) {
    $ready=$false;$seen=@{}
    foreach($line in Get-Content -LiteralPath $Path){
        if($line -eq 'g_test_objective ready version=1'){$ready=$true;continue}
        if($line -notlike 'sv_test_objective *' -and $line -notlike 'g_test_objective *'){continue}
        if($line -notmatch '^sv_test_objective spawncount=(-?\d+) server_frame=(\d+) g_test_objective version=1 map=([a-z][a-z0-9_]*) game_frame=(\d+) alive=(\d+) pending=(\d+)$'){throw 'Invalid or unbound native combat objective'}
        $r=[pscustomobject]@{spawncount=[int]$Matches[1];frame=[int]$Matches[2];map=$Matches[3];game_frame=[int]$Matches[4];alive=[int]$Matches[5];pending=[int]$Matches[6]}
        $key="$($r.spawncount)/$($r.map)/$($r.frame)";if($seen.ContainsKey($key)){throw 'Duplicate native combat objective frame'};$seen[$key]=$true
        $r
    }
    if(!$ready){throw 'Native combat objective telemetry not confirmed'}
}
