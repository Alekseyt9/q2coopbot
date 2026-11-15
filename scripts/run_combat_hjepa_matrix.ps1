[CmdletBinding()]
param([Parameter(Mandatory)][string]$Plan,[Parameter(Mandatory)][string]$Models,
 [Parameter(Mandatory)][string]$OutputRoot,[int]$Port=35600)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$planPath=(Resolve-Path -LiteralPath $Plan).Path
$modelsPath=(Resolve-Path -LiteralPath $Models).Path
$root=[IO.Path]::GetFullPath($OutputRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh matrix output required'}
Push-Location $repo
try {
 & go run ./cmd/q2episode --verify-plan $planPath --root $repo
 if($LASTEXITCODE){throw 'Generated plan verification failed'}
 $schedule=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json
 $variants=@(Get-Content -LiteralPath $modelsPath -Raw|ConvertFrom-Json)
 if(@($variants.name|Sort-Object -Unique).Count -ne $variants.Count){throw 'Duplicate variant'}
 foreach($v in $variants){
  if($v.name -notmatch '^[a-z0-9-]+$' -or $v.mode -notin 'rules','learned'){throw 'Invalid variant'}
  if($v.mode -eq 'learned'){
   $v.model=(Resolve-Path -LiteralPath $v.model).Path
   if((Get-FileHash -LiteralPath $v.model).Hash.ToLowerInvariant() -ne $v.sha256){throw 'Frozen model changed'}
  }
 }
 foreach($task in $schedule.tasks){
  if($task.seeds.Count -lt 1 -or $task.seeds.Count -gt 4 -or $task.episode.recipe.loadout -notin 'blaster','machinegun','shotgun','weapons','weapons-scarce','weapons-ssg','weapons-grenades-recovery' -or $task.instances.Count -ne $task.seeds.Count){throw 'Matrix requires 1..4 generated supported inventory instances per task'}
 }
 $maxWorkers=($schedule.tasks|ForEach-Object {$_.seeds.Count}|Measure-Object -Maximum).Maximum
 $batchSize=[int][math]::Floor(16/$maxWorkers)
 . "$PSScriptRoot/harness_manifest.ps1"
 . "$PSScriptRoot/combat_harness_bundle.ps1"
 $bundle=New-HarnessBinaryBundle $repo
 New-Item -ItemType Directory -Path $root | Out-Null
 Copy-Item -LiteralPath $planPath -Destination "$root/plan.json"
 $variants|ConvertTo-Json -Depth 12|Set-Content -LiteralPath "$root/models.json" -Encoding utf8NoBOM
 $jobs=@();$index=0
 foreach($v in $variants){for($i=0;$i -lt $schedule.tasks.Count;$i++){
  $task=$schedule.tasks[$i]
  $fixture="$root/fixtures-$i.json"
  if(!(Test-Path -LiteralPath $fixture)){@{instances=@($task.instances)}|ConvertTo-Json -Depth 30|Set-Content -LiteralPath $fixture -Encoding utf8NoBOM}
  $out="$root/$($v.name)/case-$i"
  $argsMap=@{Workers=$task.seeds.Count;EpisodesPerWorker=1;Synchronous=$true;Loadout=$task.episode.recipe.loadout;Mixed=[bool]$task.episode.recipe.mixed;ReleaseGameFrame=100;
   GameFrames=$task.episode.game_frames;Seed=[int]$task.seeds[0];Port=$Port+$maxWorkers*($index%$batchSize);Skill=$task.episode.skill;
   CombatMode=$v.mode;StopOnGoal=$true;StopOnFirstDeath=$true;RewardConfig=(Join-Path $repo $task.episode.recipe.reward_config);
   GeneratedFixtures=$fixture;OutputRoot=$out;BinaryBundle=$bundle}
  if($v.mode -eq 'learned'){$argsMap.ProviderFile=$v.model}
  if($v.contract){$argsMap.ExecutionContract=$v.contract}
  if($v.contract -and $v.contract -ne 'legacy_v1'){$argsMap.CheckedWallSlide=$true}
  $jobFile="$root/job-$index.clixml";$argsMap|Export-Clixml -LiteralPath $jobFile
  $jobs+=@{index=$index;variant=$v.name;task=$i;args=$jobFile;out=$out};$index++
 }}
 $hostExe=(Get-Process -Id $PID).Path
 # At most sixteen native games, with 1..4 clients per job.
 for($start=0;$start -lt $jobs.Count;$start+=$batchSize){
  $active=@()
  foreach($j in @($jobs|Select-Object -Skip $start -First $batchSize)){
   $launcher="$root/launch-$($j.index).ps1"
   # CLIXML carries structured arguments; paths below are fixed launcher literals.
   $launch="`$ErrorActionPreference='Stop'`n`$a=Import-Clixml -LiteralPath '"+$j.args.Replace("'","''")+"'`n& '"+(Join-Path $PSScriptRoot 'run_learned_combat_baseline.ps1').Replace("'","''")+"' @a`n"
   [IO.File]::WriteAllText($launcher,$launch)
   $proc=Start-Process -FilePath $hostExe -ArgumentList @('-NoProfile','-File',('"'+$launcher+'"')) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$root/job-$($j.index).stdout.log" -RedirectStandardError "$root/job-$($j.index).stderr.log"
   $active+=@{process=$proc;job=$j}
  }
  foreach($a in $active){$a.process.WaitForExit();if($a.process.ExitCode -ne 0){throw "Matrix job failed: $($a.job.index)"}}
  foreach($a in $active){
   $r=Get-Content -LiteralPath "$($a.job.out)/report.json" -Raw|ConvertFrom-Json
   if(!$r.provenance_valid -or !$r.capture_complete -or @($r.results|Where-Object {!$_.capture_valid}).Count){throw 'Invalid native matrix capture'}
  }
  Write-Output "Completed matrix jobs $start through $([math]::Min($start+$batchSize-1,$jobs.Count-1))"
 }
 foreach($v in $variants){if($v.mode -eq 'learned' -and (Get-FileHash -LiteralPath $v.model).Hash.ToLowerInvariant() -ne $v.sha256){throw 'Model changed during matrix'}}
 @{state='complete';jobs=$jobs;source_plan_sha256=(Get-FileHash $planPath).Hash.ToLowerInvariant();models=$variants}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath "$root/complete.json" -Encoding utf8NoBOM
} finally {Pop-Location}
