[CmdletBinding()]
param([int[]]$Timescales=@(2,1),[int]$Port=29540,[ValidateSet('near','far')][string]$Range='near',[ValidateRange(1,10)][int]$Repeats=1,[string]$RuntimeRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/runtime/q2go-damage'),[string]$OutputRoot='',[string]$ProjectileFixture='')
$ErrorActionPreference='Stop'
. "$PSScriptRoot/check_projectile_combat.ps1"
. "$PSScriptRoot/read_projectile_ledger.ps1"
. "$PSScriptRoot/read_projectile_fixture.ps1"
$fixture=$null;if($ProjectileFixture){$fixture=Read-ProjectileFixture $ProjectileFixture}
if(!$OutputRoot){$OutputRoot=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/artifacts/projectile-comparison-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$runs=@();$pairs=@()
foreach($repeat in 1..$Repeats) {
foreach($mode in @('projectile_blaster','projectile_hyper')) {
    foreach($baseline in @($false,$true)) {
        $label=if($baseline){'baseline'}else{'lead'}
        $trial=Join-Path $OutputRoot "repeat-$repeat/$mode-$label"
        & "$PSScriptRoot/run_speed_trial.ps1" -RuntimeRoot $RuntimeRoot -CombatMoveTrial -WeaponSwitchTrial $mode -ProjectileComparison -ProjectileRange $Range -ProjectileFixture $ProjectileFixture -DisableProjectileLead:$baseline -DamageTrace -SynchronizedStart -UnlimitedLoopbackRate -GameFrames 160 -Timescales $Timescales -Port $Port -OutputRoot $trial
        foreach($run in @(Get-Content (Join-Path $trial 'summary.json') -Raw|ConvertFrom-Json)) {
            $rows=@(Get-Content $run.trace_jsonl|ConvertFrom-Json)
            $damage=@(Read-DamageEvents $run.combat_log)
            $detail=Measure-ProjectileCombat $rows $damage $mode $baseline (Read-CombatStart $run.combat_log) $Range $fixture
            $shots=@(Read-ProjectileLedger $run.combat_log $damage)
            $mod=if($mode -eq 'projectile_hyper'){10}else{1}
            $weaponShots=@($shots|Where-Object {$_.attacker -eq $rows[0].self_entity -and $_.mod -eq $mod})
            if(!$weaponShots.Count){throw 'No actual projectiles emitted'}
            if($mod -eq 10 -and $weaponShots.Count -ne $detail.cells_spent){throw 'Projectile count does not match observed Cells expenditure'}
            $metrics=@(Measure-ProjectileLedger $weaponShots $rows[0].self_entity)[0]
            if($metrics.unresolved -or $metrics.unknown_freed){throw 'Comparison ended without complete projectile outcomes'}
            $detail|Add-Member -NotePropertyName projectiles -NotePropertyValue $metrics
            $runs += [pscustomobject]@{fixture=$fixture;repeat=$repeat;range=$Range;timescale=$run.timescale;detail=$detail;trace=$run.trace_jsonl;server_log=$run.combat_log}
            $runs|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'runs.json')
        }
    }
    foreach($scale in $Timescales) {
        $lead=@($runs|Where-Object {$_.repeat -eq $repeat -and $_.timescale -eq $scale -and $_.detail.mode -eq $mode -and !$_.detail.baseline})[0].detail
        $base=@($runs|Where-Object {$_.repeat -eq $repeat -and $_.timescale -eq $scale -and $_.detail.mode -eq $mode -and $_.detail.baseline})[0].detail
        $matched=$lead.initial_trajectory -ceq $base.initial_trajectory
        $noRegression=Test-ProjectileNoRegression $lead $base
        $pairs += [pscustomobject]@{repeat=$repeat;range=$Range;mode=$mode;timescale=$scale;accepted=($matched -and ($Range -ne 'far' -or $noRegression));matched_initial_trajectory=$matched;no_regression=$noRegression;lead=$lead;baseline=$base;scope='descriptive_pair_no_statistical_superiority_claim'}
        $pairs|ConvertTo-Json -Depth 8|Set-Content (Join-Path $OutputRoot 'report.json')
    }
}
}
$pairs | Select-Object mode,timescale,matched_initial_trajectory,@{n='lead_damage';e={$_.lead.health_damage}},@{n='baseline_damage';e={$_.baseline.health_damage}},@{n='lead_cells';e={$_.lead.cells_spent}},@{n='baseline_cells';e={$_.baseline.cells_spent}}
if(@($pairs|Where-Object {!$_.accepted}).Count){throw 'Comparison rejected: setup differs or far-range damage/time/ammo regressed'}
