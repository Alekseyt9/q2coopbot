param([Parameter(Mandatory=$true)][string]$Trace)
$ErrorActionPreference='Stop'
$rows=@(Get-Content $Trace | ForEach-Object {ConvertFrom-Json $_})
$event=@($rows | Where-Object {$_.pickup.class -eq 'weapon_shotgun' -and $_.pickup.state -eq 'confirmed' -and $_.frame -eq $_.pickup.end_frame -and $_.pickup.after -gt $_.pickup.before})
if(!$event.Count){throw 'No confirmed shotgun pickup'}
$after=@($rows | Where-Object {$_.frame -ge $event[0].frame -and $_.inventory_known -and $_.inventory_age_frames -le 20 -and @($_.inventory | Where-Object {$_.name -eq 'Shells' -and $_.count -ge 30}).Count})
if(!$after.Count){throw 'Shell boxes not collected (weapon alone gives only 10 shells)'}
if($rows[-1].goal -ne 'cover_teammate' -or $rows[-1].health -le 0){throw 'No safe return to teammate'}
Write-Output 'PASS: shotgun, shell supplies and return to teammate'
