param([Parameter(Mandatory=$true)][string]$Trace)
$ErrorActionPreference = 'Stop'
$rows = @(Get-Content $Trace | ForEach-Object { ConvertFrom-Json $_ } | Where-Object { $_.weapon -like '*/v_machn/*' })
$streak=0; $maximum=0; $pauses=0; $lastFrame=-1
foreach($row in $rows) {
    if($row.frame -eq $lastFrame){continue}
    if($row.frame -ne $lastFrame+1){$streak=0}
    $lastFrame=$row.frame
    if($row.sent_command.Buttons -band 1){$streak++; $maximum=[Math]::Max($maximum,$streak)}else{$streak=0}
    if($row.arbitration.limit_reason -eq 'machinegun_burst_pause') {
        if($row.sent_command.Buttons -band 1){throw 'Attack remained pressed during pause'}
        $pauses++
    }
}
$ammoUsed=if($rows.Count){$rows[0].ammo-($rows | Measure-Object ammo -Minimum).Minimum}else{0}
$accepted=$maximum -gt 0 -and $maximum -le 3 -and $pauses -ge 4 -and $ammoUsed -ge 3
[pscustomobject]@{accepted=$accepted;max_attack_frames=$maximum;pause_frames=$pauses;ammo_used=$ammoUsed;trace=$Trace}
if(!$accepted){throw 'Machinegun bursts were not confirmed'}
