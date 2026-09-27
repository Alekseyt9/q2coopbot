[CmdletBinding()]
param([Parameter(Mandatory)][string]$Trace)
$ErrorActionPreference='Stop'
# This checker is for the base1 FriendlyFireTrial fixture (zero pitch delta).
# It checks transmitted aiming, not damage attribution or projectile hits.
$rows=@(Get-Content -LiteralPath $Trace | ConvertFrom-Json)
$samples=@($rows | ForEach-Object {
    $r=$_
    if ($r.map -eq 'base1' -and $r.health -gt 0 -and ($r.sent_command.Buttons -band 1) -and $r.arbitration.aim_source -eq 'enemy') {
        $targets=@($r.enemies | Where-Object {$_.class -eq 'monster_infantry' -and [math]::Abs($_.origin[0]-96) -lt 1 -and [math]::Abs($_.origin[1]+200) -lt 1})
        if ($targets.Count -eq 1 -and @($r.enemies | Where-Object clear_shot -eq $true).Count -eq 1 -and $targets[0].clear_shot) {
            $e=$targets[0]
            if ($e.solid -in @(8290,4194)) {
                $z=if ($e.solid -eq 4194) {-8} else {22}
                $dx=$e.origin[0]-$r.self[0]; $dy=$e.origin[1]-$r.self[1]
                $angle=-[math]::Round([math]::Atan2($e.origin[2]+$z-$r.self[2]-22,[math]::Sqrt($dx*$dx+$dy*$dy))*65536/(2*[math]::PI))
                [pscustomobject]@{frame=$r.frame;id=$e.id;duck=($e.solid -eq 4194);error=[math]::Abs($r.sent_command.Pitch-$angle)}
            }
        }
    }
})
$cycle=$false
foreach ($id in @($samples.id | Select-Object -Unique)) {
    $s=@($samples | Where-Object id -eq $id)
    $duck=@($s | Where-Object duck | Select-Object -First 1)
    if ($duck.Count -and @($s | Where-Object {!$_.duck -and $_.frame -lt $duck[0].frame}).Count -and @($s | Where-Object {!$_.duck -and $_.frame -gt $duck[0].frame}).Count) {$cycle=$true}
}
$bad=@($samples | Where-Object error -gt 1)
$blocked=@($rows | Where-Object {$_.arbitration.limit_reason -eq 'friendly_line_of_fire' -and !($_.sent_command.Buttons -band 1)})
$passed=$cycle -and $samples.Count -gt 0 -and $bad.Count -eq 0 -and $blocked.Count -gt 0
[pscustomobject]@{accepted=$passed;scope='observed_bbox_and_sent_pitch_not_hits';samples=$samples.Count;duck_samples=@($samples|Where-Object duck).Count;pitch_errors=$bad.Count;standing_duck_standing=$cycle;friendly_blocks=$blocked.Count;trace=$Trace}
if (!$passed) {throw 'Crouch aiming check failed: require standing/duck/standing, correct pitch and friendly-fire hold.'}

