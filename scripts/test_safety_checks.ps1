param([Parameter(Mandatory)][string]$GrenadeTrace,[Parameter(Mandatory)][string]$GrenadeSummary,[Parameter(Mandatory)][string]$PickupTrace)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_hand_grenade_guard.ps1"
. "$PSScriptRoot/check_low_health_pickup.ps1"
$grenade=@(Get-Content $GrenadeTrace | ConvertFrom-Json)
$summary=Get-Content $GrenadeSummary -Raw | ConvertFrom-Json
$pickup=@(Get-Content $PickupTrace | ConvertFrom-Json)
Assert-HandGrenadeGuard $grenade $summary | Out-Null
Assert-LowHealthPickup $pickup | Out-Null
foreach($case in @('held','consumed','missing_hand','no_combat','false_confirmation','preowned','damage','enemy','no_rejoin')) {
    $rows=@(($grenade | ConvertTo-Json -Depth 40 -Compress)|ConvertFrom-Json)
    $p=@(($pickup | ConvertTo-Json -Depth 40 -Compress)|ConvertFrom-Json)
    switch($case) {
        'held' { ($rows|Where-Object weapon -Match '/v_handgr/'|Select-Object -First 1).sent_command.Buttons=1 }
        'consumed' { ($rows[-1].inventory|Where-Object name -EQ 'Grenades').count=4 }
        'missing_hand' { foreach($r in $rows){$r.weapon='Blaster'} }
        'no_combat' { foreach($r in $rows){$r.sent_command.Buttons=0} }
        'false_confirmation' { foreach($r in $p){if($r.pickup){$r.pickup.state='approach'}} }
        'preowned' { foreach($r in $p){if($r.inventory_known){$r.inventory+= [pscustomobject]@{id=9;name='Super Shotgun';count=1}}} }
        'damage' { $p[-1].health=37 }
        'enemy' { $p[-1]|Add-Member -Force -NotePropertyName enemies -NotePropertyValue @([pscustomobject]@{class='monster_soldier'}) }
        'no_rejoin' { $p[-1].goal='collect_item' }
    }
    $rejected=$false
    try {
        if($case -in @('held','consumed','missing_hand','no_combat')) {Assert-HandGrenadeGuard $rows $summary|Out-Null}
        else {Assert-LowHealthPickup $p|Out-Null}
    } catch {$rejected=$true}
    if(!$rejected){throw "Negative control accepted: $case"}
}
Write-Host 'PASS: native baselines and 9 negative safety controls'
