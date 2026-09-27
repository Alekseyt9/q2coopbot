$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'check_ammo_yield.ps1')
$expected=@{class='ammo_shells';name='Shells';weapon='Shotgun';origin=@(928,24,-176.875);gain=10}
foreach ($case in @('valid','unknown-bot','no-weapon','wrong-item','unknown-actor','stale-actor','small-gain','wrong-place','dead-actor','changed-map','restarted-pickup')) {
    $rows=@(@{frame=65;map='base1';health=100;inventory_known=$true;inventory_age_frames=0;inventory=@(@{name='Shotgun';count=1});resource_yield=@{entity=274;class='ammo_shells';reason='teammate_closer'};pickups=@(@{id=274;class='ammo_shells';origin=@(928,24,-176.875)})})
    $actor=@(
        @{frame=65;map='base1';health=100;inventory_known=$true;inventory_age_frames=0;inventory=@();self=@(928,64,-167.875)},
        @{frame=92;map='base1';health=100;inventory_known=$true;inventory_age_frames=0;inventory=@(@{name='Shells';count=10});self=@(928,24,-167.875)}
    )
    switch ($case) {
        'unknown-bot' {$rows[0].inventory_known=$false}
        'no-weapon' {$rows[0].inventory=@()}
        'wrong-item' {$rows[0].pickups[0].origin=@(992,24,-176.875)}
        'unknown-actor' {$actor[0].inventory_known=$false}
        'stale-actor' {$actor[1].inventory_age_frames=21}
        'small-gain' {$actor[1].inventory[0].count=9}
        'wrong-place' {$actor[1].self=@(0,0,0)}
        'dead-actor' {$actor[0].health=0}
        'changed-map' {$actor[1].map='base2'}
        'restarted-pickup' {$rows+=@{frame=70;pickup=@{entity=274;state='approach'}}}
    }
    $failed=$false
    try {$stage=Assert-AmmoYield $rows $actor $expected} catch {$failed=$true}
    if ($failed -ne ($case -ne 'valid')) {throw "Unexpected result: $case"}
    if ($case -eq 'valid' -and $stage -ne 92) {throw 'Wrong acceptance stage'}
}
Write-Host 'PASS: ammo yield evidence and 10 negative controls'
