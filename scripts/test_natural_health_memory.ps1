$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'check_natural_health_memory.ps1')
function New-Evidence {
    $item=@{id=271;class='item_health';origin=@(1184,416,-40.875)}
    $target=@(1184,416,-31.75)
    @(
        @{frame=90;map='base1';health=40;goal='follow_teammate';pickups=@($item)},
        @{frame=100;map='base1';health=40;goal='recover_health';goal_point=$target;pickups=@();resource_memory=@(@{item=$item;state='unknown';attempted=$true})},
        @{frame=101;map='base1';health=40;goal='recover_health';goal_point=$target;pickups=@($item)},
        @{frame=110;map='base1';health=50;goal='recover_health';goal_point=$target;self=$target;pickups=@()}
    )
}
foreach ($case in @('valid','masked','visible-choice','never-seen','retained-goal','wrong-height','not-reobserved','dead','changed-map','no-health-gain','healed-elsewhere')) {
    $rows=New-Evidence
    switch ($case) {
        'masked' { $rows[0].test_health_masked=$true }
        'visible-choice' { $rows[1].pickups=$rows[0].pickups }
        'never-seen' { $rows[0].pickups=@() }
        'retained-goal' { $rows[0].goal='recover_health';$rows[0].goal_point=$rows[1].goal_point }
        'wrong-height' { $rows[1].goal_point=@(1184,416,100) }
        'not-reobserved' { $rows[2].pickups=@() }
        'dead' { $rows[2].health=0 }
        'changed-map' { $rows[2].map='base2' }
        'no-health-gain' { $rows[2].health=50 }
        'healed-elsewhere' { $rows[3].self=@(0,0,0) }
    }
    $failed=$false
    try { $frame=Assert-NaturalHealthMemory $rows 50 } catch { $failed=$true }
    if ($failed -ne ($case -ne 'valid')) { throw "Unexpected acceptance result: $case" }
    if ($case -eq 'valid' -and $frame -ne 110) { throw 'Wrong healing stage' }
}
Write-Host 'PASS: natural memory evidence and 10 false-positive controls'
