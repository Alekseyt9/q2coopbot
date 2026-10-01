[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report,[switch]$Blocked,[switch]$SurfaceContact)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_armed_grenade.ps1"
$items=@(Get-Content $Report -Raw|ConvertFrom-Json)
if(!$items.Count -or @($items|Where-Object {!$_.accepted}).Count){throw 'Accepted native report required'}
$bot=Get-ChildItem $items[0].artifacts -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
$actor=Get-ChildItem $items[0].artifacts -Filter '*human-trace.jsonl'|Select-Object -First 1
$original=@(Get-Content $bot.FullName|ConvertFrom-Json)
$actorRows=@(Get-Content $actor.FullName|ConvertFrom-Json)
$null=Assert-GrenadeGeometry $original $actorRows -Blocked:$Blocked -SurfaceContact:$SurfaceContact
$results=@()
$cases=@('missing-geometry','post-bounce-certified','authorized','wrong-clear-prefix','wrong-stop','stale-frame')
if($Blocked){$cases+=@('missing-bounce-bound','collapsed-bounce-bound','bounce-authorized','bounce-certified')}
if($SurfaceContact){$cases+=@('missing-surface','wrong-plane','wrong-contact','surface-authorized','wrong-surface-seed')}
foreach($case in $cases) {
    $rows=@($original|ConvertTo-Json -Depth 100|ConvertFrom-Json)
    foreach($row in $rows) {
        $envelope=$row.grenade_prediction.early_envelope
        $geometry=$envelope.geometry
        if(!$geometry){continue}
        switch($case) {
            'missing-geometry' {$envelope.PSObject.Properties.Remove('geometry')}
            'post-bounce-certified' {$geometry.post_bounce_certified=$true}
            'authorized' {$geometry.authorized=$true}
            'wrong-clear-prefix' {$geometry.static_clear_seconds=1}
            'wrong-stop' {$geometry|Add-Member -NotePropertyName stop_seconds -NotePropertyValue 0.4 -Force}
            'stale-frame' {$envelope.frame--}
            'missing-bounce-bound' {$geometry.PSObject.Properties.Remove('bounce_envelope')}
            'collapsed-bounce-bound' {if($geometry.bounce_envelope){foreach($range in $geometry.bounce_envelope.ranges){$range.min=@(0,0,0);$range.max=@(0,0,0)}}}
            'bounce-authorized' {if($geometry.bounce_envelope){$geometry.bounce_envelope.authorized=$true}}
            'bounce-certified' {if($geometry.bounce_envelope){$geometry.bounce_envelope.geometry_certified=$true}}
            'missing-surface' {$geometry.PSObject.Properties.Remove('static_surface_contact')}
            'wrong-plane' {if($geometry.static_surface_contact){$geometry.static_surface_contact.normal=@(0,1,0)}}
            'wrong-contact' {if($geometry.static_surface_contact){$geometry.static_surface_contact.min[0]=0}}
            'surface-authorized' {if($geometry.static_surface_bounce_envelope){$geometry.static_surface_bounce_envelope.authorized=$true}}
            'wrong-surface-seed' {if($geometry.static_surface_bounce_envelope){$geometry.static_surface_bounce_envelope.ranges[0].min[0]=0}}
        }
    }
    $rejected=$false;$reason=''
    try{$null=Assert-GrenadeGeometry $rows $actorRows -Blocked:$Blocked -SurfaceContact:$SurfaceContact}catch{$rejected=$true;$reason=$_.Exception.Message}
    $results+=[pscustomobject]@{case=$case;rejected=$rejected;reason=$reason}
}
$results|ConvertTo-Json -Depth 6|Set-Content (Join-Path (Split-Path $Report -Parent) 'geometry-negative-controls.json') -Encoding utf8
if(@($results|Where-Object {!$_.rejected}).Count){throw 'Damaged geometry certificate accepted'}
Write-Host "Geometry: $($results.Count)/$($results.Count) negative controls rejected"
