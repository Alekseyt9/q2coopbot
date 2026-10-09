[CmdletBinding()]
param([switch]$Apply,[switch]$PruneExports,[switch]$LzxStreams,[ValidateRange(1,8)][int]$CompressionWorkers=4,
 [string]$Python='F:/src/strat/.venv-gpu/Scripts/python.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$maintenance=Join-Path $repo 'workspace/build/artifact-maintenance'
New-Item -ItemType Directory -Path $maintenance -Force|Out-Null
$root=(Resolve-Path -LiteralPath (Join-Path $repo 'workspace/artifacts')).Path.TrimEnd('\','/')
$boundary=$root+[IO.Path]::DirectorySeparatorChar
if((Get-Item -LiteralPath $root).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Artifact root must not be a junction'}
$active=@(Get-CimInstance Win32_Process|Where-Object {
 $_.Name -match '^(q2ded|q2coopbot.*|q2ppo-data|q2combat-export|python.*)\.exe$' -or
 ($_.Name -eq 'pwsh.exe' -and $_.ProcessId -ne $PID -and $_.CommandLine -match 'run_registered_combat|process_combat_architecture|ppo_recurrent.py|ppo_combat.py')
})
if($active.Count){throw 'Stop or finish combat capture/export/training before maintenance'}
$tag=Get-Date -Format 'yyyyMMdd-HHmmss'
$planPath=Join-Path $maintenance "export-cleanup-$tag.json"
if($PruneExports){
 & $Python "$PSScriptRoot/find_rebuildable_combat_exports.py" --root $root --out $planPath
 if($LASTEXITCODE){throw 'Export inventory failed'}
 $plan=Get-Content -LiteralPath $planPath -Raw|ConvertFrom-Json
}else{
 $plan=@{root=$root;files=@();exports=@();bytes=0;scope='Transparent NTFS compression; no stream pruning requested'}
 $plan|ConvertTo-Json|Set-Content -LiteralPath $planPath -Encoding utf8NoBOM
}
if(!$Apply){Write-Output "Dry run: $($plan.files.Count) rebuildable streams, $($plan.bytes) bytes; NTFS compression available with -Apply";return}
$freeBefore=(Get-Volume -DriveLetter F).SizeRemaining
$removed=0L
if($PruneExports){
 foreach($entry in $plan.files){
  $path=(Resolve-Path -LiteralPath $entry.path).Path
  if(!$path.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Cleanup target escaped artifact root'}
  for($parent=[IO.DirectoryInfo]::new((Split-Path $path -Parent));$parent -and $parent.FullName.StartsWith($root,[StringComparison]::OrdinalIgnoreCase);$parent=$parent.Parent){
   if($parent.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Cleanup refuses junction ancestors'}
  }
  if((Get-Item -LiteralPath $path).Length -ne $entry.bytes){throw 'Cleanup candidate changed'}
  Remove-Item -LiteralPath $path
  $removed+=$entry.bytes
 }
 foreach($folder in $plan.exports){
  @{version='pruned_rebuildable_ppo_streams_v1';date=$tag;plan=$planPath;scope=$plan.scope;rebuild='Re-export the retained native batch with its frozen q2ppo-data binary and initial weights; merge again. Training checkpoints remain usable for fresh rollouts.'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $folder 'cleanup.json') -Encoding utf8NoBOM
 }
}
$log=Join-Path $maintenance "ntfs-compression-$tag.log"
& "$env:SystemRoot/System32/compact.exe" /C /I /Q $root *> $log
if($LASTEXITCODE){throw "NTFS root compression failed; inspect $log"}
$directories=@(Get-ChildItem -LiteralPath $root -Directory|Where-Object {!($_.Attributes -band [IO.FileAttributes]::ReparsePoint)})
$compactResults=@($directories|ForEach-Object -Parallel {
 $taskLog=Join-Path $using:maintenance ("compact-"+$using:tag+'-'+$_.Name+'.log')
 if($using:LzxStreams){
  & "$env:SystemRoot/System32/compact.exe" /C "/S:$($_.FullName)" /EXE:LZX /F /I /Q '*.jsonl' *> $taskLog
 }else{
  & "$env:SystemRoot/System32/compact.exe" /C "/S:$($_.FullName)" /I /Q *> $taskLog
 }
 @{directory=$_.FullName;exit_code=$LASTEXITCODE;log=$taskLog}
} -ThrottleLimit $CompressionWorkers)
$compactResults|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $maintenance "compression-results-$tag.json") -Encoding utf8NoBOM
if(@($compactResults|Where-Object exit_code -ne 0).Count){throw 'Some NTFS compression workers failed; inspect compression-results JSON'}
$result=@{version='combat_artifact_maintenance_v1';artifact_root=$root;plan=$planPath;compression_log=$log;lzx_streams=[bool]$LzxStreams;pruned_logical_bytes=$removed;free_before=$freeBefore;free_after=(Get-Volume -DriveLetter F).SizeRemaining;scope='Native inputs/reports/checkpoints retained. NTFS directory compression inherited; optional WOF LZX for closed JSONL. Logical size differs from size on disk.'}
$result|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $maintenance "artifact-maintenance-$tag.json") -Encoding utf8NoBOM
$result|ConvertTo-Json
