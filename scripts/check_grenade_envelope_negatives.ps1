[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report,[ValidateSet('teammate','target')][string]$Kind='teammate')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_armed_grenade.ps1"
$items=@(Get-Content $Report -Raw|ConvertFrom-Json)
if(!$items.Count -or @($items|Where-Object {!$_.accepted}).Count){throw 'Accepted native report required'}
$file=Get-ChildItem $items[0].artifacts -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
$original=@(Get-Content $file.FullName|ConvertFrom-Json)
$null=Assert-GrenadeEarlyEnvelope $original -Kind $Kind
$results=@()
foreach($case in @('missing-envelope','stale-frame','authorized','geometry-certified','missing-contact-range','inverted-bounds')) {
    $rows=@($original|ConvertTo-Json -Depth 100|ConvertFrom-Json)
    foreach($row in $rows) {
        $envelope=$row.grenade_prediction.early_envelope
        if(!$envelope){continue}
        switch($case) {
            'missing-envelope' {$row.grenade_prediction.PSObject.Properties.Remove('early_envelope')}
            'stale-frame' {$envelope.frame--}
            'authorized' {$envelope.authorized=$true}
            'geometry-certified' {$envelope.geometry_certified=$true}
            'missing-contact-range' {$envelope.PSObject.Properties.Remove('contacts')}
            'inverted-bounds' {foreach($contact in $envelope.contacts){$contact.min[0]=$contact.max[0]+1}}
        }
    }
    $rejected=$false;$reason=''
    try{$null=Assert-GrenadeEarlyEnvelope $rows -Kind $Kind}catch{$rejected=$true;$reason=$_.Exception.Message}
    $results+=[pscustomobject]@{case=$case;rejected=$rejected;reason=$reason}
}
$out=Join-Path (Split-Path $Report -Parent) 'envelope-negative-controls.json'
$results|ConvertTo-Json -Depth 6|Set-Content $out -Encoding utf8
if(@($results|Where-Object {!$_.rejected}).Count){throw 'Damaged envelope accepted'}
Write-Host "Envelope: $($results.Count)/$($results.Count) negative controls rejected"
