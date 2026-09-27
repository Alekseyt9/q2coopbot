[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29410,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
if (!$OutputRoot) {$OutputRoot=Join-Path $repo ('workspace/artifacts/weapon-economy-pair-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
New-Item -ItemType Directory -Path $OutputRoot -Force|Out-Null
$results=@()
function Count-Stock($row,$name) { $items=@($row.inventory|Where-Object name -eq $name);if ($items.Count) {return [int]$items[0].count};return 0 }
foreach ($mode in @('economy_pair_weak','economy_pair_heavy')) {
    $out=Join-Path $OutputRoot $mode
    & "$PSScriptRoot/run_speed_trial.ps1" -CombatMoveTrial -WeaponSwitchTrial $mode -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 220 -Timescales $Timescales -Port $Port -OutputRoot $out
    foreach ($scale in $Timescales) {
        $path=@(Get-ChildItem $out -Filter "scale-$scale-port-*-trace.jsonl"|Where-Object Name -NotLike '*human*')[0].FullName
        $rows=@(Get-Content $path|ConvertFrom-Json)
        $heavy=$mode -eq 'economy_pair_heavy'
        $class=if ($heavy) {'monster_tank'} else {'monster_soldier'}
        $initial=@($rows|Where-Object {
            $_.arbitration.limit_reason -ne 'test_weapon_setup' -and $_.weapon -eq 'Blaster' -and $_.inventory_known -and $_.inventory_age_frames -le 1 -and
            (Count-Stock $_ 'Cells') -eq 100 -and (Count-Stock $_ 'Slugs') -eq 10 -and (Count-Stock $_ 'Shells') -eq 20 -and
            (Count-Stock $_ 'HyperBlaster') -eq 1 -and (Count-Stock $_ 'Railgun') -eq 1 -and (Count-Stock $_ 'Shotgun') -eq 1 -and
            @($_.enemies|Where-Object { $_.class -eq $class -and $_.clear_shot -and [math]::Abs($_.origin[0]-96) -lt 16 -and [math]::Abs($_.origin[1]+200) -lt 16 }).Count
        }|Select-Object -First 1)
        $accepted=$false;$reason='initial_inventory_or_target_missing';$metrics=$null
        if ($initial.Count) {
            $target=@($initial[0].enemies|Where-Object { $_.class -eq $class -and $_.clear_shot -and [math]::Abs($_.origin[0]-96) -lt 16 -and [math]::Abs($_.origin[1]+200) -lt 16 })[0].id
            $fire=@($rows|Where-Object { $_.frame -ge $initial[0].frame -and ($_.sent_command.Buttons -band 1) -and @($_.enemies|Where-Object { $_.id -eq $target -and $_.clear_shot }).Count -and (($heavy -and $_.weapon -like '*/v_hyperb/*') -or (!$heavy -and $_.weapon -eq 'Blaster')) }|Select-Object -First 1)
            $death=@($rows|Where-Object { $_.frame -gt $initial[0].frame -and @($_.defeated|Where-Object { $_.id -eq $target -and $_.class -eq $class }).Count -and $_.inventory_known -and $_.inventory_age_frames -le 1 }|Select-Object -First 1)
            $reason='firing_or_observed_death_missing'
            if ($fire.Count -and $death.Count -and $death[0].frame -gt $fire[0].frame) {
                $window=@($rows|Where-Object { $_.frame -ge $initial[0].frame -and $_.frame -le $death[0].frame })
                $requests=@($window|Where-Object weapon_request)
                $cells=100-(Count-Stock $death[0] 'Cells');$slugs=10-(Count-Stock $death[0] 'Slugs');$shells=20-(Count-Stock $death[0] 'Shells')
                $selection=if ($heavy) {@($requests|Where-Object { $_.weapon_request -eq 'use HyperBlaster' -and $_.weapon_reason -eq 'heavy_target' -and $_.frame -le $fire[0].frame }).Count -eq 1} else {$requests.Count -eq 0}
                $accepted=$selection -and $requests.Count -le 3 -and $slugs -eq 0 -and $shells -eq 0 -and (($heavy -and $cells -gt 0 -and $cells -lt 100) -or (!$heavy -and $cells -eq 0)) -and !@($window|Where-Object { $_.health -le 0 -or $_.map -ne 'base1' }).Count
                if (@($rows|Where-Object { $_.frame -ge $death[0].frame -and @($_.enemies|Where-Object id -eq $target).Count }).Count) {$accepted=$false}
                $reason=if ($accepted) {'accepted'} else {'selection_spend_or_death_filter_failed'}
                $metrics=@{entity=$target;initial_frame=$initial[0].frame;first_selected_weapon_fire=$fire[0].frame;death_observed_frame=$death[0].frame;game_seconds_to_death=($death[0].frame-$initial[0].frame)/10.0;cells_used=$cells;slugs_used=$slugs;shells_used=$shells;requests=$requests.Count}
            }
        }
        $results += [pscustomobject]@{mode=$mode;timescale=$scale;accepted=$accepted;reason=$reason;metrics=$metrics;trace=$path}
        $results|ConvertTo-Json -Depth 6|Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
$results|Format-Table mode,timescale,accepted,reason
if (@($results|Where-Object {!$_.accepted}).Count) {throw 'Matched inventory economy trial failed'}
