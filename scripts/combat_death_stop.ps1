# Supervisor-only candidate; the exporter must verify ordered native terminal and reward.
function Get-CombatFirstLifeDeathReceipt($Row, $Events, $Release, [string]$Map='base1') {
    if($Map -notmatch '^[a-z][a-z0-9_]{0,31}$' -or !$Row -or !$Release -or !$Row.combat_policy.observation){return $null}
    $identity=$Row.combat_policy.observation.identity
    if($null -eq $Row.health -or $null -eq $Row.combat_policy.observation.health -or $Row.health -ne $Row.combat_policy.observation.health){return $null}
    if($Row.health -gt 0 -or $identity.life -ne 1 -or $Row.connection -ne 1 -or $Row.map -ne $Map){return $null}
    if($Row.spawncount -ne $Release.spawncount -or $identity.spawncount -ne $Release.spawncount -or $identity.connection -ne 1 -or $identity.map -ne $Map -or $identity.actor -ne $Row.self_entity -or $identity.frame -ne $Row.observation_frame){return $null}
    if($Row.self_entity -le 0 -or $Row.observation_frame -le $Release.frame){return $null}
    $deaths=@($Events | Where-Object {$_.spawncount -eq $Release.spawncount -and $_.map -eq $Map -and $_.target -eq $Row.self_entity -and $_.target_class -eq 'player' -and $_.killed} | Sort-Object frame)
    if(!$deaths.Count){return $null}
    $death=$deaths[0]
    if($death.frame -lt $Release.frame -or $death.frame -gt $Row.observation_frame -or $death.health_before -le 0 -or $death.health_after -gt 0){return $null}
    [pscustomobject]@{version='combat_first_life_death_stop_v1';reason='combat_first_life_death';map=$Map;spawncount=$Release.spawncount;actor=$Row.self_entity;death_frame=[int]$death.frame;observed_frame=[int]$Row.observation_frame;health=[int]$Row.health}
}
