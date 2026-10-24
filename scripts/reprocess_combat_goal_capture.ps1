# Reuse ordinary finalization on preserved raw data. No server/client is launched.
param([Parameter(Mandatory)][string]$CaptureRoot,[Parameter(Mandatory)][string]$Plan,
      [Parameter(Mandatory)][int]$TaskIndex,[Parameter(Mandatory)][int]$SeedIndex,
      [Parameter(Mandatory)][int]$Port,[Parameter(Mandatory)][string]$OutputRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. "$PSScriptRoot/generated_combat_fixture.ps1"
$artifacts=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts'))+[IO.Path]::DirectorySeparatorChar
$original=(Resolve-Path -LiteralPath $CaptureRoot).Path
$target=[IO.Path]::GetFullPath($OutputRoot)
if(!$original.StartsWith($artifacts,[StringComparison]::OrdinalIgnoreCase) -or !$target.StartsWith($artifacts,[StringComparison]::OrdinalIgnoreCase)){throw 'Artifact roots required'}
if(Test-Path -LiteralPath $target){throw 'Fresh derived output required'}
$originalReport=Get-Content -LiteralPath (Join-Path $original 'report.json') -Raw|ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $original 'manifest.json') -Raw|ConvertFrom-Json
if(!$originalReport.provenance_valid -or $originalReport.results.Count -ne 1){throw 'Original provenance required'}
if(!$originalReport.capture_complete -and $originalReport.results[0].error -ne 'Unverified goal stop'){throw 'Only goal-boundary finalization failures admitted'}
$schedule=Get-Content -LiteralPath $Plan -Raw|ConvertFrom-Json
$task=$schedule.tasks[$TaskIndex];$Seed=[int]$task.seeds[$SeedIndex];$sample=$task.instances[$SeedIndex]
if($Seed -ne $originalReport.results[0].seed -or $sample.engine_seed -ne $Seed){throw 'Original fixture/seed differs'}
if($manifest.provider -eq 'learned' -and $manifest.model_weights_sha256.ToLowerInvariant() -ne $schedule.model_sha256){throw 'Original weights differ'}
$oldWorker=$originalReport.results[0].root
$pins=@{}
foreach($p in @((Join-Path $original 'report.json'),(Join-Path $original 'manifest.json'),$Plan)+@(Get-ChildItem -LiteralPath $oldWorker -File|ForEach-Object FullName)){
    $pins[$p]=(Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant()
}
$null=New-Item -ItemType Directory -Path $target
$out=Join-Path $target 'worker-0-episode-0';$null=New-Item -ItemType Directory -Path $out
foreach($file in @(Get-ChildItem -LiteralPath $oldWorker -File)){
    if($file.Name -in @('combat-first-life.json','reset-expectation.json')){Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $out $file.Name)}
    else{$null=New-Item -ItemType HardLink -Path (Join-Path $out $file.Name) -Target $file.FullName}
}
$null=New-Item -ItemType Junction -Path (Join-Path $out 'runtime') -Target (Join-Path $oldWorker 'runtime')
Copy-Item -LiteralPath (Join-Path $original 'manifest.json') -Destination (Join-Path $target 'manifest.json')
$exporter=Join-Path $original 'q2combat-export.exe'
if((Get-FileHash -LiteralPath $exporter).Hash.ToLowerInvariant() -ne $manifest.exporter_sha256.ToLowerInvariant()){throw 'Frozen exporter differs'}
$null=New-Item -ItemType HardLink -Path (Join-Path $target 'q2combat-export.exe') -Target $exporter
foreach($file in @(Get-ChildItem -LiteralPath $original -File)){
    if($file.Name -eq 'q2coopbot.exe' -or $file.Name -like 'fixture-*.json' -or $file.Name -like 'provider-worker-*.json'){
        $pins[$file.FullName]=(Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        $null=New-Item -ItemType HardLink -Path (Join-Path $target $file.Name) -Target $file.FullName
    }
}
$Synchronous=$true;$StopOnGoal=$true;$StopOnFirstDeath=$true;$Feedback=$false;$TrainingMonsterHealth=0
$GameFrames=[int]$manifest.game_frames;$ReleaseGameFrame=[int]$manifest.release_game_frame;$Skill=[int]$manifest.skill
$Loadout=$manifest.loadout;$CombatMode=$manifest.provider;$RewardConfig=Join-Path $repo $task.episode.recipe.reward_config
if((Get-FileHash -LiteralPath $RewardConfig).Hash.ToLowerInvariant() -ne $manifest.reward_config_sha256.ToLowerInvariant()){throw 'Original reward differs'}
$episodeFixture='standard';$episodeMixed=[bool]$task.episode.recipe.mixed;$captureMap=$sample.map
$originalConfig=Get-Content -LiteralPath (Join-Path $oldWorker 'bot-config.json') -Raw|ConvertFrom-Json
$episodeProvider=if($CombatMode -ne 'rules'){$originalConfig.combat.provider_file}else{$null}
$episodeProviderHash=if($episodeProvider){(Get-FileHash -LiteralPath $episodeProvider).Hash}else{$null}
if($episodeProvider){$pins[$episodeProvider]=$episodeProviderHash.ToLowerInvariant()}
if($originalReport.capture_complete -and $episodeProviderHash -ne $originalReport.results[0].provider_config_sha256){throw 'Original per-episode provider differs'}
$worker=0;$episode=0;$index=0
$process=[pscustomobject]@{ExitCode=$originalReport.results[0].worker_exit_code}
$timer=[pscustomobject]@{Elapsed=[pscustomobject]@{TotalSeconds=$originalReport.results[0].cold_episode_seconds}}
# Execute the exact ordinary validation/export block, replacing only parallel
# scope prefixes. This avoids implementing a relaxed recovery validator.
$baseline=Get-Content -LiteralPath "$PSScriptRoot/run_learned_combat_baseline.ps1" -Raw
$baseline=$baseline.Replace("`r`n","`n")
$start=$baseline.IndexOf('        try{'+"`n"+'            $feedbackReport=$null')
$finish=$baseline.IndexOf("`n    }`n"+'} -ThrottleLimit $Workers)',$start)
if($start -lt 0 -or $finish -le $start){throw 'Ordinary finalization block not found'}
$post=$baseline.Substring($start,$finish-$start).Replace('$using:','$')
if($post -match 'Start-Process|Stop-Process|Remove-Item'){throw 'Reprocessing may not launch clients or delete evidence'}
$pins|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $target 'original-inputs.json') -Encoding utf8NoBOM
$values=@(& ([scriptblock]::Create($post)))
$result=@($values|Where-Object {$_.PSObject.Properties.Name -contains 'capture_valid'})
if($result.Count -ne 1 -or !$result[0].capture_valid -or !$result[0].dispatch_valid -or !$result[0].seed_confirmed){
    $values|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $target 'failure.json') -Encoding utf8NoBOM
    throw 'Ordinary reprocessing did not accept capture'
}
foreach($p in $pins.Keys){if((Get-FileHash -LiteralPath $p).Hash.ToLowerInvariant() -ne $pins[$p]){throw 'Original evidence changed'}}
$derived=$originalReport.PSObject.Copy()
$derived.results=@($result[0]);$derived.capture_complete=$true;$derived.usable_captures=1
$derived|ConvertTo-Json -Depth 25|Set-Content -LiteralPath (Join-Path $target 'report.json') -Encoding utf8NoBOM
@{state='complete';version='combat_goal_capture_reprocessing_v1';original_root=$original;inputs=$pins;
    raw_evidence_unchanged=$true;new_games=$false;training_performed=$false;neural_execution=$false;
    original_capture_complete=$originalReport.capture_complete;goal=$result[0].goal_stop;
    script_sha256=(Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant();
    finalizer_sha256=(Get-FileHash -LiteralPath "$PSScriptRoot/run_learned_combat_baseline.ps1").Hash.ToLowerInvariant();
    report_sha256=(Get-FileHash -LiteralPath (Join-Path $target 'report.json')).Hash.ToLowerInvariant()}|
    ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $target 'reprocessing.json') -Encoding utf8NoBOM
'Preserved capture reprocessed without replaying the battle'
