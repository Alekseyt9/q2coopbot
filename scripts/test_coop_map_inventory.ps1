param([Parameter(Mandatory)][string]$Trace,[Parameter(Mandatory)][string]$ServerLog,[Parameter(Mandatory)][string]$ObserverLog)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_coop_map_inventory.ps1"
$original=@(Get-Content $Trace|ConvertFrom-Json)
$server=[IO.File]::ReadAllText((Resolve-Path $ServerLog))
$observer=[IO.File]::ReadAllText((Resolve-Path $ObserverLog))
Assert-CoopMapInventory $original $server $observer|Out-Null
foreach($case in @('lost_weapon','lost_ammo','new_connection','no_death','native_reconnect','no_acquisition')) {
    $rows=@(($original|ConvertTo-Json -Depth 40 -Compress)|ConvertFrom-Json)
    $native=$server
    switch($case) {
        'lost_weapon' { $rows[-1].inventory=@($rows[-1].inventory|Where-Object name -NE 'Super Shotgun') }
        'lost_ammo' { ($rows[-1].inventory|Where-Object name -EQ 'Shells').count=0 }
        'new_connection' { $rows[-1].connection=2 }
        'no_death' { foreach($r in $rows){if($r.health -le 0){$r.health=100}} }
        'native_reconnect' { $native+="`nGoCoopMate connected`n" }
        'no_acquisition' { foreach($r in $rows){if($r.pickup){$r.pickup.state='approach'}} }
    }
    $rejected=$false
    try {Assert-CoopMapInventory $rows $native $observer|Out-Null} catch {$rejected=$true}
    if(!$rejected){throw "Negative control accepted: $case"}
}
Write-Host 'PASS: native baseline and 6 negative inventory controls'
