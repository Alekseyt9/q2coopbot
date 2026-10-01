[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_armed_grenade.ps1"
$items=@(Get-Content $Report -Raw|ConvertFrom-Json)
if(!$items.Count -or @($items|Where-Object {!$_.accepted}).Count){throw 'Accepted native report required'}
$run=$items[0].artifacts
$botFile=Get-ChildItem $run -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
$actorFile=Get-ChildItem $run -Filter '*human-trace.jsonl'|Select-Object -First 1
$original=@(Get-Content $botFile.FullName|ConvertFrom-Json)
$actor=@(Get-Content $actorFile.FullName|ConvertFrom-Json)
$null=Assert-GrenadeMovingTarget $original $actor
$results=@()
foreach($case in @('missing-contact','stale-frame','wrong-identity','static-target','attack','health-mask','missing-risk')) {
    $rows=@($original|ConvertTo-Json -Depth 100|ConvertFrom-Json)
    switch($case) {
        'missing-contact' {foreach($row in $rows){foreach($sample in $row.grenade_prediction.samples){$sample.PSObject.Properties.Remove('target_contact')}}}
        'stale-frame' {foreach($row in $rows){foreach($motion in $row.grenade_prediction.target_motion){$motion.frame--}}}
        'wrong-identity' {foreach($row in $rows){foreach($motion in $row.grenade_prediction.target_motion){$motion.entity+=10000}}}
        'static-target' {foreach($row in $rows){foreach($enemy in $row.enemies){$enemy.origin[1]=-128}}}
        'attack' {$row=$rows|Where-Object {$_.frame -lt 350 -and $_.weapon -match '/v_handgr/'}|Select-Object -Last 1;$row.sent_command.Buttons=1}
        'health-mask' {$rows[0]|Add-Member -NotePropertyName test_health_masked -NotePropertyValue $true -Force}
        'missing-risk' {foreach($row in $rows){foreach($sample in $row.grenade_prediction.samples){$sample.risk='unproven'}}}
    }
    $rejected=$false;$reason=''
    try{$null=Assert-GrenadeMovingTarget $rows $actor}catch{$rejected=$true;$reason=$_.Exception.Message}
    $results+=[pscustomobject]@{case=$case;rejected=$rejected;reason=$reason}
}
$out=Join-Path (Split-Path $Report -Parent) 'negative-controls.json'
$results|ConvertTo-Json -Depth 6|Set-Content $out -Encoding utf8
if(@($results|Where-Object {!$_.rejected}).Count){throw 'Damaged trace accepted'}
Write-Host "Moving target negative controls: $out ($($results.Count)/$($results.Count) rejected)"
