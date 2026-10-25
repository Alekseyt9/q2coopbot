$ErrorActionPreference='Stop'
. "$PSScriptRoot/generated_combat_fixture.ps1"
$work=Join-Path (Split-Path $PSScriptRoot -Parent) ('workspace/build/generated-boundary-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work|Out-Null
$path=Join-Path $work 'isolated.ent'
$retained=@'
{"classname" "worldspawn" "message" "campaign"}
{"classname" "func_door" "model" "*3" "target" "exit"}
{"classname" "trigger_teleport" "target" "tele_dest"}
{"classname" "info_player_start" "origin" "0 0 0"}
{"classname" "monster_berserk" "origin" "100 0 0"}
'@
$exits=@'
{"classname" "trigger_changelevel" "map" "waste3" "model" "*8"}
{"map" "waste1" "classname" "target_changelevel" "targetname" "t38"}
'@
[IO.File]::WriteAllText($path,$retained+"`n"+$exits,[Text.Encoding]::ASCII)
$proof=Set-GeneratedCombatWorldBoundary $path
$actual=[IO.File]::ReadAllText($path)
if($proof.version -ne 'isolated_combat_no_changelevel_v2' -or $proof.removed_changelevels -ne 2){throw 'Missing exit-removal receipt'}
if($actual -ne $retained+"`n"+"`n"){throw 'Entities other than level exits changed'}
if($proof.entities_sha256 -ne (Get-FileHash $path).Hash.ToLowerInvariant()){throw 'Receipt does not bind entity contents'}
$again=Set-GeneratedCombatWorldBoundary $path
if($again.removed_changelevels -ne 0 -or $again.entities_sha256 -ne $proof.entities_sha256){throw 'Boundary operation is not idempotent'}
'Generated combat world boundary: passed (two exits, retained actors/movers/teleport, hash, no-exit case)'
