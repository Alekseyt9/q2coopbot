[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_grenade_auto.ps1"
$items=@(Get-Content $Report -Raw|ConvertFrom-Json)
if($items.Count -ne 4 -or @($items|Where-Object {!$_.accepted}).Count){throw 'Accepted native report required'}
$results=@()
foreach($item in @($items|Group-Object scenario|ForEach-Object {$_.Group[0]})) {
    $bot=Get-ChildItem $item.artifacts -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
    $actor=Get-ChildItem $item.artifacts -Filter '*human-trace.jsonl'|Select-Object -First 1
    $original=@(Get-Content $bot.FullName|ConvertFrom-Json);$actorRows=@(Get-Content $actor.FullName|ConvertFrom-Json)
    $reject=$item.scenario -match 'friend-reject'
    $cases=if($reject){@('missing-inventory','unsafe-selection','unsafe-projectile')}else{@('missing-start','held-attack','missing-release','missing-consumption','damaged-teammate','forced-arming')}
    foreach($case in $cases) {
        $rows=@($original|ConvertTo-Json -Depth 100|ConvertFrom-Json);$actors=@($actorRows|ConvertTo-Json -Depth 100|ConvertFrom-Json)
        foreach($row in $rows|Where-Object {$_.frame -ge 140}) {
            switch($case) {
                'missing-inventory' {$row.inventory=@()}
                'unsafe-selection' {$row.weapon_request='use Grenades'}
                'unsafe-projectile' {$row.projectiles=@(@{id=90;class='hand_grenade';origin=@(0,0,0)})}
                'missing-start' {if($row.arbitration.limit_reason -eq 'hand_grenade_auto_start'){$row.arbitration.limit_reason='none'}}
                'held-attack' {if($row.gun_frame -eq 11){$row.sent_command.Buttons=1}}
                'missing-release' {if($row.arbitration.limit_reason -eq 'hand_grenade_release'){$row.arbitration.limit_reason='none'}}
                'missing-consumption' {$row.inventory=@()}
                'forced-arming' {$row.arbitration.limit_reason='test_grenade_arming'}
            }
        }
        if($case -eq 'damaged-teammate'){foreach($row in $actors){$row.health=1}}
        $failed=$false;$reason=''
        try{$null=Assert-GrenadeAuto $rows $actors -Reject:$reject}catch{$failed=$true;$reason=$_.Exception.Message}
        $results+=@{scenario=$item.scenario;case=$case;rejected=$failed;reason=$reason}
    }
}
$results|ConvertTo-Json -Depth 6|Set-Content (Join-Path (Split-Path $Report -Parent) 'negative-controls.json') -Encoding utf8
if(@($results|Where-Object {!$_.rejected}).Count){throw 'Damaged automatic grenade evidence accepted'}
Write-Host "Automatic grenade controls: $($results.Count)/$($results.Count) rejected"
