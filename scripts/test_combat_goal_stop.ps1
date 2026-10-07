$ErrorActionPreference='Stop'
. "$PSScriptRoot/combat_goal_stop.ps1"
. "$PSScriptRoot/read_damage_events.ps1"
$row=[pscustomobject]@{observation_frame=170;spawncount=42;self_entity=1;health=56;map='base1';connection=1;combat_policy=@{observation=@{identity=@{life=1}}}}
$context=@{spawncount=42;frame=98};$classes=@('monster_parasite','monster_gunner')
$events=@(
    [pscustomobject]@{spawncount=42;map='base1';frame=147;attacker=1;target=352;target_class='monster_parasite';attacker_class='player';mod=4;killed=$true;health_before=7},
    [pscustomobject]@{spawncount=42;map='base1';frame=169;attacker=1;target=343;target_class='monster_gunner';attacker_class='player';mod=4;killed=$true;health_before=7},
    [pscustomobject]@{spawncount=42;map='base1';frame=155;attacker=1;target=243;target_class='func_explosive';attacker_class='player';mod=4;killed=$true;health_before=2}
)
if(!(Get-CombatGoalReceipt $row $events $context $classes)){throw 'Real native victory not detected'}
if(Get-CombatGoalReceipt $row @($events | Where-Object target_class -ne 'monster_gunner') $context $classes){throw 'Missing target accepted'}
if(Get-CombatGoalReceipt $row @($events | Where-Object {$_.health_before -le 0}) $context $classes){throw 'Corpse damage accepted'}
$early=$row.PSObject.Copy();$early.observation_frame=169
if(Get-CombatGoalReceipt $early $events $context $classes){throw 'Stopped before closing observation'}
$row.health=0
if(Get-CombatGoalReceipt $row $events $context $classes){throw 'Dead bot accepted'}
$row.health=56;$row.connection=2
if(Get-CombatGoalReceipt $row $events $context $classes){throw 'Reconnect accepted'}
$row.connection=1
if(Get-CombatGoalReceipt $row $events @{spawncount=$row.spawncount+1;frame=98} $classes){throw 'Wrong world accepted'}
if(Get-CombatGoalReceipt $row $events @{spawncount=$row.spawncount;frame=170} $classes){throw 'Preparation kills accepted'}
$death=[pscustomobject]@{spawncount=42;map='base1';frame=150;target=1;target_class='player';killed=$true}
if(Get-CombatGoalReceipt $row @($events+$death) $context $classes){throw 'Death followed by respawn accepted'}
$row.combat_policy.observation.identity.life=2
if(Get-CombatGoalReceipt $row $events $context $classes){throw 'Second life accepted'}
'Goal stop regression checks passed'
