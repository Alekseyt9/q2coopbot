[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_rail_damage.ps1"
$runs=@(Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json)
$count=0
foreach($run in $runs) {
    $rows=@(Get-Content $run.trace | ConvertFrom-Json)
    $events=@(Read-DamageEvents $run.server_log)
    $null=Assert-RailDamage $rows $events $run.mode
    $cases=if($run.mode -eq 'rail_precision'){@('attacker','target','mod','generation','early','no_damage')}else{@('unsafe_contact')}
    foreach($case in $cases) {
        $changed=@($events | ConvertTo-Json -Depth 6 | ConvertFrom-Json)
        if($case -eq 'unsafe_contact') {
            $first=@($rows|Where-Object {$_.weapon -like '*/v_rail/*'})[0]
            $changed+= [pscustomobject]@{spawncount=$first.spawncount;map='base1';frame=$first.frame;attacker=$first.self_entity;mod=11}
        }else{
            foreach($e in $changed) {
                switch($case) {
                    'attacker' {$e.attacker=999}
                    'target' {$e.target=999}
                    'mod' {$e.mod=1}
                    'generation' {$e.spawncount=0}
                    'early' {$e.frame=$rows[0].frame}
                    'no_damage' {$e.live_health_damage=0}
                }
            }
        }
        $rejected=$false
        try{$null=Assert-RailDamage $rows $changed $run.mode}catch{$rejected=$true}
        if(!$rejected){throw "False damage acceptance: $case scale=$($run.timescale)"}
        $count++
    }
}
"PASS: $count invalid damage attributions rejected"
