[CmdletBinding()]
param([Parameter(Mandatory)][string]$Runs)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_combat.ps1"
$count=0
foreach($run in @(Get-Content $Runs -Raw|ConvertFrom-Json)) {
    $fixture=$run.fixture
    $lines=Get-Content $run.trace
    $events=@(Read-DamageEvents $run.server_log)
    $start=Read-CombatStart $run.server_log
    $range=if($run.detail.range){$run.detail.range}else{'near'}
    $null=Measure-ProjectileCombat @($lines|ConvertFrom-Json) $events $run.detail.mode $run.detail.baseline $start $range $fixture
    $cases=@('missing_start','weapon_switch','foreign_damage','wrong_aim','early_fire','airborne_start')
    if($range -eq 'far'){$cases+=@('near_start','no_crossing')}
    if($fixture){$cases+=@('wrong_light','foreign_monster','actor_position')}
    foreach($case in $cases) {
        $rows=@($lines|ConvertFrom-Json)
        $changed=@($events|ConvertTo-Json -Depth 5|ConvertFrom-Json)
        switch($case) {
            'wrong_light' {$rows[0].sent_command.Light=0}
            'foreign_monster' {foreach($r in $rows){if($r.frame -eq $start){$r.enemies+=@{id=999;class='monster_soldier'}}}}
            'actor_position' {foreach($r in $rows){if($r.frame -eq $start){$r.teammate[0]+=32}}}
            'missing_start' {$rows=@($rows|Where-Object frame -ne $start)}
            'weapon_switch' {$rows[0]|Add-Member -NotePropertyName weapon_request -NotePropertyValue 'use Shotgun' -Force}
            'foreign_damage' {foreach($e in $changed){$e.attacker=999}}
            'wrong_aim' {foreach($r in $rows){if($r.arbitration.aim_point){$r.arbitration.aim_point[0]+=50}}}
            'early_fire' {foreach($r in $rows){if($r.frame -eq $start){$r.sent_command.Buttons=1}}}
            'airborne_start' {foreach($r in $rows){if($r.frame -eq $start){$r.on_ground=$false}}}
            'near_start' {foreach($r in $rows){if($r.frame -ge $start -and $r.frame -lt $start+10){$r.self[0]=1136;$r.self[1]=256}}}
            'no_crossing' {foreach($r in $rows){if($r.frame -ge $start -and $r.frame -lt $start+10){foreach($e in @($r.enemies|Where-Object id -eq $run.detail.target)){$e.origin[0]=1088;$e.origin[1]=328}}}}
        }
        $rejected=$false
        try{$null=Measure-ProjectileCombat $rows $changed $run.detail.mode $run.detail.baseline $start $range $fixture}catch{$rejected=$true}
        if(!$rejected){throw "Invalid projectile comparison accepted: $case"}
        $count++
    }
    $zero=Measure-ProjectileCombat @($lines|ConvertFrom-Json) @() $run.detail.mode $run.detail.baseline $start $range $fixture
    if($zero.health_damage -ne 0 -or $zero.kills -ne 0 -or $null -ne $zero.kill_after_start_frames){throw 'No-hit run incorrectly counted as success'}
}
"PASS: $count invalid comparisons rejected; no-hit controls report zero damage and no kill"
