$ErrorActionPreference='Stop'
. "$PSScriptRoot/combat_goal_stop.ps1"
$row=[pscustomobject]@{observation_frame=21;spawncount=42;self_entity=1;health=80;map='boss2';connection=1;combat_policy=@{observation=@{identity=@{life=1}}}}
$release=@{spawncount=42;frame=10;require_objective=$true}
$kills=@([pscustomobject]@{spawncount=42;map='boss2';frame=20;attacker=1;target=50;target_class='monster_jorg';attacker_class='player';mod=4;killed=$true;health_before=5})
$state=[pscustomobject]@{spawncount=42;frame=21;map='boss2';game_frame=23;alive=0;pending=1}
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)){throw 'Jorg death animation/delayed Makron accepted as win'}
$state.pending=0;$state.alive=1
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)){throw 'Living successor or revived monster accepted'}
$state.alive=0
$goal=Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)
if(!$goal -or $goal.version -ne 'combat_goal_stop_v2'){throw 'Closed native objective rejected'}
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2'){throw 'Missing native truth accepted'}
$state.frame=20
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)){throw 'Stale native truth accepted'}
$state.frame=22
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)){throw 'Future zero state accepted'}
$state.frame=21;$state.spawncount++
if(Get-CombatGoalReceipt $row $kills $release @('monster_jorg') 'boss2' @($state)){throw 'Wrong generation accepted'}
'Native objective goal checks passed'
