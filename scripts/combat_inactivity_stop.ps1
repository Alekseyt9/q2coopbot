# Supervisor-only native events. A step end closes the preceding frame.
function Update-CombatActivity($State, [string[]]$Lines, $Release, [int]$Actor, [int]$IdleFrames) {
    foreach ($line in $Lines) {
        if ($State.deadline) { break }
        if ($line -match '^sv_test_weapon_fire spawncount=(-?\d+) server_frame=(\d+) g_test_weapon_fire version=1 actor=(\d+)$') {
            if ([int]$Matches[1] -eq $Release.spawncount -and [int]$Matches[2] -ge $Release.frame -and [int]$Matches[3] -eq $Actor) {
                $State.last = [Math]::Max($State.last, [int]$Matches[2])
            }
        } elseif ($line -match '^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) .* target=(\d+) .* health_before=(-?\d+) health_after=(-?\d+) .* target_class=(\w+) ') {
            if ([int]$Matches[1] -eq $Release.spawncount -and [int]$Matches[2] -ge $Release.frame -and
                [int]$Matches[4] -gt 0 -and [int]$Matches[5] -lt [int]$Matches[4] -and
                ([int]$Matches[3] -eq $Actor -or $Matches[6].StartsWith('monster_'))) {
                $State.last = [Math]::Max($State.last, [int]$Matches[2])
            }
        } elseif ($line -match '^sv_test_step version=1 phase=end spawncount=(-?\d+) frame=(\d+) seq=\d+ actor=(\d+)$') {
            if ([int]$Matches[1] -eq $Release.spawncount -and [int]$Matches[3] -eq $Actor -and
                [int]$Matches[2] - $State.last -ge $IdleFrames) {
                $State.deadline = [int]$Matches[2]
            }
        }
    }
}

function Get-CombatInactivityReceipt($Activity, $Rows, $Release, [int]$Actor, [string]$Map, [int]$IdleFrames) {
    if (!$Activity.deadline) { return }
    foreach ($row in $Rows) {
        if ($row.observation_frame -ne $Activity.deadline -or $row.health -le 0 -or $row.self_entity -ne $Actor -or
            $row.map -ne $Map -or $row.spawncount -ne $Release.spawncount -or $row.connection -ne 1 -or
            $row.combat_policy.observation.identity.life -ne 1) { continue }
        return [pscustomobject]@{version='combat_inactivity_stop_v1';reason='combat_inactivity';map=$Map;spawncount=$Release.spawncount;
            actor=$Actor;observed_frame=$Activity.deadline;last_activity_frame=$Activity.last;idle_frames=$IdleFrames}
    }
}
