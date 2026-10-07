[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunRoot,[Parameter(Mandatory)][string]$Output)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Output){throw 'Fresh diagnostic output required'}
$root=(Resolve-Path -LiteralPath $RunRoot).Path
$training=Get-Content -LiteralPath (Join-Path $root 'report.json') -Raw|ConvertFrom-Json
if($training.state -ne 'complete'){throw 'Completed registry experiment required'}
. "$PSScriptRoot/read_damage_events.ps1"
. "$PSScriptRoot/read_projectile_ledger.ps1"
$results=@()
foreach($evaluation in $training.evaluations){
    $binding=Get-Content (Join-Path $evaluation.root 'report.json') -Raw|ConvertFrom-Json
    if($binding.state -ne 'complete'){throw 'Incomplete evaluation'}
    foreach($record in $binding.records){
        $reportPath=Join-Path $record.artifacts 'report.json'
        if((Get-FileHash $reportPath).Hash.ToLowerInvariant() -ne $record.report_sha256){throw 'Capture report changed'}
        $batch=Get-Content $reportPath -Raw|ConvertFrom-Json
        if(!$batch.capture_complete -or !$batch.provenance_valid){throw 'Invalid native capture'}
        foreach($episode in $batch.results){
            $rows=@(Get-Content (Join-Path $episode.root 'bot.jsonl')|ForEach-Object {$_|ConvertFrom-Json}|Where-Object {$_.combat_policy.observation.identity.life -eq 1}|Sort-Object observation_frame -Unique)
            $controlled=@($rows|Where-Object {$_.combat_policy.selection.owner -eq 'provider'})
            $recoil=@($controlled|Where-Object {$_.combat_policy.observation.kick_angles_degrees -and [math]::Abs($_.combat_policy.observation.kick_angles_degrees[0]) -gt 0})
            $bullets=0
            for($i=1;$i -lt $rows.Count;$i++){
                $a=$rows[$i-1].combat_policy.observation;$b=$rows[$i].combat_policy.observation
                if($a.weapon -match 'Machinegun|v_machn' -and $b.weapon -match 'Machinegun|v_machn' -and $b.identity.frame -eq $a.identity.frame+1 -and $a.ammo -gt $b.ammo){$bullets+=[int]($a.ammo-$b.ammo)}
            }
            $native=@(Read-DamageEvents (Join-Path $episode.root 'server.log'))
            $projectiles=@(Read-ProjectileLedger (Join-Path $episode.root 'server.log') $native|Where-Object {$_.attacker -eq $episode.first_life.bot_entity -and $_.spawn_frame -ge $episode.first_life.start_frame -and $_.spawn_frame -le $episode.first_life.end_frame})
            $results+=@{model=$evaluation.model;label=$evaluation.label;family=$record.episode_id;seed=$episode.seed;native_start_confirmed=$episode.generated_start.confirmed;goal_complete=[bool]$episode.goal_stop;first_life=$episode.first_life.damage;provider_frames=$controlled.Count;nonzero_kick_frames=$recoil.Count;max_abs_kick_pitch=$(if($recoil.Count){($recoil|ForEach-Object {[math]::Abs($_.combat_policy.observation.kick_angles_degrees[0])}|Measure-Object -Maximum).Maximum}else{0});observed_bullets_decrease=$bullets;projectile_ledger=@(Measure-ProjectileLedger $projectiles $episode.first_life.bot_entity);root=$episode.root}
        }
    }
}
@{version=1;run_root=$root;report_sha256=(Get-FileHash (Join-Path $root 'report.json')).Hash.ToLowerInvariant();scope='First observed life. Kick angles contain weapon and incoming-damage kicks; provider-frame kick presence does not prove compensation skill or identify its cause. Bullet decrease includes all controllers in the first life and is an observed lower bound; hitscan contacts are not shot accuracy. Projectile ledger only covers native projectile weapons.';results=$results}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $Output -Encoding utf8NoBOM
Write-Output $Output
