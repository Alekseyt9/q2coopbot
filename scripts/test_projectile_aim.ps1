[CmdletBinding()]
param([Parameter(Mandatory)][string]$Trace,[ValidateSet('projectile_blaster','projectile_hyper')][string]$Mode='projectile_hyper')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_aim.ps1"
$lines=Get-Content -LiteralPath $Trace
$null=Assert-ProjectileAim @($lines | ConvertFrom-Json) $Mode
foreach($case in @('wrong_point','wrong_time','wrong_weapon','missing_observation','no_fire','wrong_entity')) {
    $rows=@($lines | ConvertFrom-Json)
    foreach($r in $rows) {
        switch($case) {
            'wrong_point' {if($r.arbitration.lead_seconds -gt 0){$r.arbitration.aim_point[1]+=30}}
            'wrong_time' {if($r.arbitration.lead_seconds -gt 0){$r.arbitration.lead_seconds*=1.5}}
            'wrong_weapon' {$r.weapon='Railgun'}
            'missing_observation' {if($r.enemies){$r.enemies=@()}}
            'no_fire' {$r.sent_command.Buttons=0}
            'wrong_entity' {if($r.arbitration.aim_entity){$r.arbitration.aim_entity=-1}}
        }
    }
    $rejected=$false
    try{$null=Assert-ProjectileAim $rows $Mode}catch{$rejected=$true}
    if(!$rejected){throw "False projectile acceptance: $case"}
}
'PASS: 6 invalid projectile traces rejected'
