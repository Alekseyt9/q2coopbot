[CmdletBinding()]
param([Parameter(Mandatory)][string]$Comparison,[Parameter(Mandatory)][string]$OutputRoot)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $OutputRoot){throw 'Fresh analysis output required'}
New-Item -ItemType Directory -Path $OutputRoot|Out-Null
. "$PSScriptRoot/read_damage_events.ps1"
. "$PSScriptRoot/read_projectile_ledger.ps1"
$comparisonPath=(Resolve-Path -LiteralPath $Comparison).Path
$comparisonData=Get-Content -LiteralPath $comparisonPath -Raw|ConvertFrom-Json
if(!$comparisonData.results.Count){throw 'Comparison has no results'}
$repo=Split-Path $PSScriptRoot -Parent
$results=@()
foreach($run in $comparisonData.results){
    if(!$run.infrastructure_ok){throw 'Cannot diagnose an unverified capture'}
    $sourceRoot=if($run.map -eq 'base1'){'combat-campaign-comparison-v1-r2-20261007'}else{'combat-campaign-base2-v2-20261007'}
    $dir=Join-Path $repo "workspace/artifacts/$sourceRoot/$($run.map)-$($run.mode)-$($run.seed)"
    $trace=Join-Path $dir 'bot.jsonl'
    $log=Join-Path $dir 'server.log'
    $rows=@()
    $reader=[IO.File]::OpenText($trace)
    try{
        while($null -ne ($line=$reader.ReadLine())){
            $r=$line|ConvertFrom-Json
            if($r.map -eq $run.map){$rows+=,$r}
        }
    }finally{$reader.Dispose()}
    if(!$rows.Count){throw 'Missing map rows'}
    $rows=@($rows|Group-Object spawncount,frame|ForEach-Object {$_.Group[0]})
    $actor=$rows[0].self_entity
    $allDamage=@(Read-DamageEvents $log)
    $shots=@(Read-ProjectileLedger $log $allDamage|Where-Object {$_.map -eq $run.map -and $_.attacker -eq $actor})
    $projectileMetrics=@(Measure-ProjectileLedger $shots $actor)
    $attacks=0;$attackNoVisible=0;$providerAttacks=0;$changed=0;$interventions=@{};$lifes=@{};$classes=@{};$deathFrames=@();$prior=$rows[0]
    foreach($r in $rows){
        $capture=$r.combat_policy
        $o=$capture.observation
        $lifeKey="$($r.spawncount):$($o.identity.life)"
        if(!$lifes.ContainsKey($lifeKey)){$lifes[$lifeKey]=@{life=$o.identity.life;spawncount=$r.spawncount;first_frame=$r.frame;last_frame=$r.frame;attack_frames=0;longest_attack_burst=0;current_burst=0}}
        $life=$lifes[$lifeKey];$life.last_frame=$r.frame
        $attack=($r.sent_command.buttons -band 1) -ne 0
        if($attack){
            $attacks++;$life.attack_frames++;$life.current_burst++
            $life.longest_attack_burst=[math]::Max($life.longest_attack_burst,$life.current_burst)
            if(!$o.enemies.Count){$attackNoVisible++}
            if($capture.selection.owner -eq 'provider'){$providerAttacks++}
        }else{$life.current_burst=0}
        if($capture.command_changed){$changed++}
        foreach($i in $capture.selection.interventions){$key="$($i.component):$($i.reason)";$interventions[$key]++}
        foreach($enemy in $o.enemies){$classes[$enemy.class]++}
        if($prior.health -gt 0 -and $r.health -le 0){$deathFrames+=@{frame=$r.frame;position=$r.self;life=$o.identity.life;previous_health=$prior.health}}
        $prior=$r
    }
    $damage=@($allDamage|Where-Object map -eq $run.map)
    $kills=@($damage|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*' -and $_.killed}).Count
    $dealt=[int](($damage|Where-Object {$_.attacker -eq $actor -and $_.target_class -like 'monster_*'}|Measure-Object live_health_damage -Sum).Sum)
    if($kills -ne $run.monsters_killed -or $dealt -ne $run.monster_health_damage -or $deathFrames.Count -ne $run.deaths){throw "Diagnostic count disagrees with verified comparison: $dir"}
    $lifeReports=@($lifes.Values|Sort-Object spawncount,life|ForEach-Object {$_.Remove('current_burst');$_})
    $report=@{map=$run.map;mode=$run.mode;seed=$run.seed;source=$dir;frames=$rows.Count;attack_frames=$attacks;attack_without_observed_enemy_frames=$attackNoVisible;provider_attack_frames=$providerAttacks;final_command_changed_frames=$changed;interventions=$interventions;observed_enemy_class_frames=$classes;deaths=$deathFrames;lives=$lifeReports;projectiles=$projectileMetrics;counts_match_original=$true}
    $report|ConvertTo-Json -Depth 12|Set-Content (Join-Path $OutputRoot "$($run.map)-$($run.mode)-$($run.seed).json") -Encoding utf8
    $results+=$report
}
$summary=@{version=1;comparison=$comparisonPath;comparison_sha256=(Get-FileHash $comparisonPath).Hash.ToLowerInvariant();scope='offline diagnostics; projectile fraction only counts native Blaster/HyperBlaster shots; no causal labels';results=$results}
if($results.Count -ne $comparisonData.results.Count){throw 'Incomplete diagnostic result set'}
$summary|ConvertTo-Json -Depth 14|Set-Content (Join-Path $OutputRoot 'report.json') -Encoding utf8
Write-Output $OutputRoot
