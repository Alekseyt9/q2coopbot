[CmdletBinding()]
param([Parameter(Mandatory)][string]$PrecisionTrace,[Parameter(Mandatory)][string]$BehindTrace)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_rail_aim.ps1"
$count=0
foreach($mode in @('rail_precision','rail_friend_behind')) {
    $path=if($mode -eq 'rail_precision'){$PrecisionTrace}else{$BehindTrace}
    $lines=Get-Content -LiteralPath $path
    $null=Assert-RailAim @($lines | ConvertFrom-Json) $mode
    $cases=if($mode -eq 'rail_precision'){@('early_fire','wrong_yaw','wrong_pitch','missing_angles','no_ammo_spent')}else{@('unsafe_fire','off_line','ammo_spent')}
    foreach($case in $cases) {
        $rows=@($lines | ConvertFrom-Json)
        $rail=@($rows | Where-Object {$_.weapon -like '*/v_rail/*' -and $_.health -gt 0 -and $_.map -eq 'base1'})
        foreach($r in $rail) {
            switch($case) {
                'early_fire' {$r.sent_command.Buttons=1}
                'wrong_yaw' {$r.sent_command.Yaw+=4096}
                'wrong_pitch' {$r.sent_command.Pitch+=4096}
                'missing_angles' {$r.PSObject.Properties.Remove('delta_angles')}
                'no_ammo_spent' {$r.ammo=10}
                'unsafe_fire' {$r.sent_command.Buttons=1}
                'off_line' {$r.teammate[1]+=200}
                'ammo_spent' {if($r.frame -gt $rail[0].frame){$r.ammo=9}}
            }
        }
        $rejected=$false
        try{$null=Assert-RailAim $rows $mode}catch{$rejected=$true}
        if(!$rejected){throw "False rail acceptance: $case"}
        $count++
    }
}
"PASS: $count invalid rail traces rejected"
