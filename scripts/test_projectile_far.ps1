[CmdletBinding()]
param([Parameter(Mandatory)][string[]]$Reports,[string]$OriginalRegressionReport='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_combat.ps1"
$count=0
foreach($path in $Reports){
    $pairs=@(Get-Content $path -Raw|ConvertFrom-Json)
    foreach($pair in $pairs){
        if($pair.range -ne 'far' -or !$pair.matched_initial_trajectory -or !(Test-ProjectileNoRegression $pair.lead $pair.baseline)){throw "Far comparison rejected: $path"}
        if($pair.lead.initial_distance -lt 400 -or $pair.baseline.initial_distance -lt 400 -or $pair.lead.crossing_frames -lt 2 -or $pair.baseline.crossing_frames -lt 2){throw 'Far geometry was not exercised'}
        $count++
    }
}
if($OriginalRegressionReport){
    $old=@(Get-Content $OriginalRegressionReport -Raw|ConvertFrom-Json|Where-Object mode -eq 'projectile_hyper')
    if(!$old.Count){throw 'Original regression missing'}
    foreach($pair in $old){if(Test-ProjectileNoRegression $pair.lead $pair.baseline){throw 'Original regression incorrectly accepted'}}
}
"PASS: $count far pairs satisfy damage/time/ammo gates; supplied original regressions rejected"
