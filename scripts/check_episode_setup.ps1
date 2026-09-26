# Shared by the registry runner and offline checks of recorded setup evidence.
function Assert-EpisodeSetup($Rows, $Acceptance) {
    foreach ($check in $Acceptance.setup_checkpoints) {
        $found = $false
        foreach ($row in $Rows) {
            if ($row.frame -lt $check.min_frame -or $row.frame -gt $check.max_frame -or !$row.on_ground -or $row.health -le 0) { continue }
            $matches = $true
            foreach ($field in @('self','teammate')) {
                if (!$check.$field) { continue }
                if (@($row.$field).Count -ne 3) { $matches = $false; break }
                for ($axis=0; $axis -lt 3; $axis++) {
                    if ([math]::Abs($row.$field[$axis]-$check.$field[$axis]) -gt 2) { $matches = $false }
                }
            }
            foreach ($expected in $check.movers) {
                $mover = @($row.movers | Where-Object model -eq $expected.model)
                if ($mover.Count -ne 1 -or @($mover[0].origin).Count -ne 3) { $matches = $false; continue }
                for ($axis=0; $axis -lt 3; $axis++) {
                    if ([math]::Abs($mover[0].origin[$axis]-$expected.origin[$axis]) -gt 1) { $matches = $false }
                }
            }
            if ($matches) { $found = $true; break }
        }
        if (!$found) { throw "Episode setup checkpoint not observed: $($check.id)" }
    }
}
