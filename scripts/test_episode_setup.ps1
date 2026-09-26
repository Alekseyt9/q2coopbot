$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'check_episode_setup.ps1')
$acceptance = @{setup_checkpoints=@(@{id='bridge'; min_frame=10; max_frame=11; self=@(1,2,3); teammate=@(4,5,6); movers=@(@{model=99; origin=@(0,-250,0)})})}
foreach ($case in @('valid','closed_bridge','hidden_bridge','wrong_floor','too_early','airborne','missing_teammate')) {
    $row = [pscustomobject]@{frame=10; on_ground=$true; health=60; self=@(1,2,3); teammate=@(4,5,6); movers=@([pscustomobject]@{model=99;origin=@(0,-250,0)})}
    switch ($case) {
        closed_bridge { $row.movers[0].origin=@(0,0,0) }
        hidden_bridge { $row.movers=@() }
        wrong_floor { $row.self[2]=-100 }
        too_early { $row.frame=9 }
        airborne { $row.on_ground=$false }
        missing_teammate { $row.teammate=$null }
    }
    $rejected=$false
    try { Assert-EpisodeSetup @($row) $acceptance } catch { $rejected=$true }
    if ($rejected -eq ($case -eq 'valid')) { throw "Incorrect checkpoint result: $case" }
}
Write-Output 'PASS: setup checkpoints reject missing or incorrect world state'
