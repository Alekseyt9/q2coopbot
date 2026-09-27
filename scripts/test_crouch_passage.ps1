[CmdletBinding()]
param([Parameter(Mandatory)][string]$Trace,[string]$TooTightTrace='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_crouch_passage.ps1"
$lines=Get-Content -LiteralPath $Trace
$null=Assert-NaturalCrouchPassage @($lines | ConvertFrom-Json)
foreach ($case in @('no_request','standing_inside','bypass','jump','death','map','gap','no_exit')) {
    $rows=@($lines | ConvertFrom-Json)
    $inside=@($rows | Where-Object {$_.frame -gt 40 -and $_.self[0] -ge -478 -and $_.self[0] -le -430})
    switch ($case) {
        'no_request' {foreach ($r in $rows) {if ($r.arbitration.skill) {$r.arbitration.skill=''}}}
        'standing_inside' {$inside[0].ducked=$false}
        'bypass' {$inside[0].self[1]=0}
        'jump' {$inside[0].sent_command.Up=200}
        'death' {$inside[0].health=0}
        'map' {$inside[0].map='base2'}
        'gap' {$rows=@($rows | Where-Object frame -ne $inside[0].frame)}
        'no_exit' {$rows=@($rows | Where-Object {$_.frame -le $inside[-1].frame})}
    }
    $rejected=$false
    try {$null=Assert-NaturalCrouchPassage $rows} catch {$rejected=$true}
    if (!$rejected) {throw "False acceptance: $case"}
}
'PASS: 8 invalid traversal traces rejected'
if ($TooTightTrace) {
    $tight=Get-Content -LiteralPath $TooTightTrace
    $null=Assert-TooTightCrouchRefusal @($tight | ConvertFrom-Json)
    foreach ($case in @('duck_request','ducked','wrong_setup','missing_frame','no_anchor','no_cover')) {
        $rows=@($tight | ConvertFrom-Json)
        $r=@($rows | Where-Object frame -eq 44)[0]
        switch ($case) {
            'duck_request' {$r.sent_command.Up=-200}
            'ducked' {$r.ducked=$true}
            'wrong_setup' {@($rows | Where-Object frame -eq 40)[0].self[1]=-35}
            'missing_frame' {$rows=@($rows | Where-Object frame -ne 44)}
            'no_anchor' {@($rows | Where-Object frame -eq 42)[0].self[1]=-42}
            'no_cover' {foreach ($row in $rows) {if ($row.goal -eq 'cover_teammate') {$row.goal='follow_teammate'}}}
        }
        $rejected=$false
        try {$null=Assert-TooTightCrouchRefusal $rows} catch {$rejected=$true}
        if (!$rejected) {throw "False tight-passage acceptance: $case"}
    }
    'PASS: 6 invalid tight-passage traces rejected'
}
