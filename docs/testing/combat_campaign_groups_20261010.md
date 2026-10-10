# Campaign ground groups — 2026-10-10

## Implementation

Added registry generator `campaign-ground-group-v1` for 2–4 distinct ground
classes: Parasite, Soldier, Infantry, Gunner. Class order is canonical; changing
CLI argument order does not change recipes or seeds. Repeated classes and
unsupported airborne/boss classes are rejected. Each additional monster has
its own start distribution. BSP checks cover grounding, static hull collision,
inline brushes and actor overlaps. Native frame-100 receipts verify every
actor's position, stock health and solid state. Full dynamic world equivalence
is not certified by these checks.

CLI: `q2episode --campaign-sites 1 --campaign-maps base3,ware2,jail1,city1
--campaign-group monster_parasite,monster_infantry,monster_gunner --out ...`.

Four registered recipes use original map geometry and Super Shotgun initially;
Machinegun, Shotgun and Blaster are owned too. Train/validation/test/confirmation
have separate seed ranges. Runtime goal completion requires native player kills
of every declared class and a subsequent living first-life observation.
The offline verifier's old two-monster limit was raised to four; a missing
native kill or an omitted receipt member still fails verification.

## First validation

`combat-group-smoke-pool-v1-20261010`: 16 independent episodes, 16-slot pool,
timescale 2, base3/ware2/jail1/city1, four validation seeds each.
Model: `combat-ssg-training-processing-v2-20261010/update/weights.json`
(SHA `cd84d88a692aa292c712c0a07957073cb4673b12523371528072175e06fe1853`).
16/16 captures usable; source remained unchanged; native starts confirmed.
Seven goals and nine deaths, 1078 received health damage, 1845 game frames.

Native log audit: 67 NPC-to-NPC damage attempts explicitly blocked by the
no-infighting fixture; zero recorded applied NPC-to-NPC damage events.
Incoming player damage is present: the fixture does not suppress NPC attacks
on the player in these captures. Audit file:
`workspace/artifacts/combat-group-native-audit-v1-20261010.json`.

Initial offline processing v1 rejected group wins because of the verifier's
two-member limit. After correction, v2 processed all 16 captures on CUDA:
1746 rows; no training on validation data. Failed v1 output is retained as a
diagnostic. `TestGoalStopRequiresEveryGroupMember` covers three/four-member
goals and rejects missing native kills and incomplete receipts.
Generator tests cover invalid compositions, canonical argument order,
geometry-checked split starts and overlapping actors. Targeted Go checks
execute geometry/evidence code, not neural numerical verification.

## Training and comparison

Prepared 32 new train episodes (eight per recipe), separate from validation,
in `combat-group-training-plan-v1-20261010.json`. Capture pool uses 16 slots
with immediate queued refill, timescale 2. Training must resume the same
SSG checkpoint on CUDA; evaluate before/after using identical validation
fixtures, seeds, frozen client/exporter and reward configuration. No promotion
or improvement claim is made until that comparison completes.

Train capture completed: 32/32 usable, 11 goals/21 deaths, 3753 game frames,
91.90 seconds wall time. All 32 cases were finalized on CUDA: 3642 rows.
Fresh before-update validation captured with the updated frozen binaries:
16/16 usable, 7 goals/9 deaths. By map: base3 3/1, ware2 2/2,
jail1 0/4, city1 2/2. Baseline pool:
`combat-group-quality-before-pool-v1-20261010`.

CUDA training completed: 10 actor steps, updates_completed=8. Weights SHA
`47fe9ecfe1cda616656c2000cae05e6dee8d91c424f9aaff4d358b727d2e536d`;
checkpoint SHA `1a737404d0cf0cacae21f34c51ae9db60f16d938dce7b0c66313dc25ceaad7c5`.
Files and completion seal:
`workspace/artifacts/combat-group-training-processing-v1-20261010/update/`.

## Paired held-out outcome

`combat-group-quality-v1-20261010.json` compares two fresh closed pools with
identical fixtures, validation seeds, client/exporter SHA and reward SHA.

| Metric (16 battles) | Before | After |
|---|---:|---:|
| Completed group goals | 7 | 8 |
| Native monster kills | 30 | 32 |
| Deaths | 9 | 8 |
| Dealt health damage | 5880 | 6124 |
| Received health damage | 1078 | 1074 |
| Game frames until episode end | 1863 | 2021 |

Goals by map: base3 3→3, ware2 2→2, jail1 0→0, city1 2→3.
Time grew 8.5%; these frames include losing fights and do not measure pure
kill speed. The win-rate change is only one battle on 16 paired episodes.
No general superiority or promotion is established; keep this checkpoint
experimental. Next: diagnose jail1 deaths from native incoming damage,
target selection and movement histories, then collect a broader fresh train
batch and evaluate on additional separated sites/seeds. Do not use this
validation corpus for gradient training.

Post-update validation processing completed on CUDA: 16/16 cases, 1957 rows,
training_performed=false, `combat-group-quality-after-processing-v1-20261010`.

Initial jail1 diagnosis (after-update validation): Parasite caused 190 of 380
received health damage, Infantry 118, Gunner 72. All three classes damage
the player. Native observed movement exists in 71/79, 80/81, 111/118 and
106/117 policy-command transitions respectively; the defeats are not explained
by the bot simply standing still throughout combat. These counters do not
prove effective evasion or movement distance. In seed 312530002 the bot dealt
162/175 HP to Gunner, 96/100 to Infantry and 98/175 to Parasite but killed none.
Hypothesis for the next training iteration: improve threat prioritization and
finishing a target while avoiding Parasite damage. Confirm with actual
target-selection/action traces before changing reward or observation contract.

## Target audit and inventory expansion

Added `scripts/audit_registered_target_choices.py`, an offline audit of actual
captured provider intent. It counts consecutive same-world switches by entity
and visible track, rather than by distance-sorted target slot. Client clear-shot
evidence is recorded, but does not prove that a switch was useless. No neural
inference or modification of gameplay is performed by the audit.

After-update jail1 captures: 395 provider frames, 253 intent changes, 245 changes
with both the previous and next target still observed and clear. Parasite is
closer than 288 units in 361 frames, selected in 137 of them. Only 150 applied
attack-command frames are present. Audit:
`combat-group-target-audit-after-v1-20261010.json`. These data motivate further
target-choice training, but do not justify treating every switch as an error.
The reward remains v8 during the inventory expansion experiment.

Added optional `--campaign-group-loadout`:
`blaster`, `machinegun`, `weapons`, `weapons-ssg` (legacy default).
The API preserves old SSG recipes/seeds. Other inventories use separate seed
domains (512 seeds per split); canonical map/site/composition identities remain.
Tests validate split/loadout separation and reject unsupported inventories.
Registered eight new descriptors on base3/ware2/jail1/city1: Machinegun-only
and MG/Shotgun/Blaster. Registry validation passes; native acceptance must be
reported separately. Blaster group generation is available but not registered
or live-validated in this experiment.

Prepared 32 fresh training episodes using `weapons` inventory, resumed behavior
weights from group update8, and a separate 32-battle paired held-out plan:
four fresh validation seeds for each of eight descriptors (weapons and SSG).
Validation offset4 avoids reusing the previous 16-battle selection sample.

`combat-group-weapons-training-pool-v1-20261010` completed 32/32 usable
captures in 90.74 seconds, 3247 game frames: five goals, 27 deaths.
Native player damage includes MOD_MACHINEGUN (4976 HP, seven kills),
MOD_SHOTGUN (2288 HP, 18 kills), MOD_BLASTER (200 HP, zero kills).
Thus owned-weapon use is exercised; this does not establish optimal weapon
selection. Both native startup and the goal/death supervision are accepted.
CUDA processing/training output:
`combat-group-weapons-training-processing-v1-20261010`.

CUDA update completed: 32 cases, 3118 rows, continuation from update8.
Weights SHA `709a1b1a912e5c6cfacf5a1cbbb969a9268d552d176367b7cec57c34d6fe61c2`;
checkpoint SHA `d10f8a19664f2988256b8b4868a81c1d94a647304de72451927766adbd308ace`.
Native before-update paired capture: 32/32 usable, weapons 1/16 goals,
SSG 8/16 goals. After-update evaluation is a separate fresh capture of
exactly the same registered fixtures and seeds; no validation gradient training.

Paired comparison `combat-group-loadout-quality-v1-20261010.json` completed
with equal client/exporter/reward SHA and identical generated starts:

| 32 held-out battles | Before | After |
|---|---:|---:|
| Group goals | 9 | 9 |
| Kills | 41 | 43 |
| Deaths | 23 | 23 |
| Dealt health damage | 9205 | 8926 |
| Received health damage | 2384 | 2288 |
| Game frames | 3369 | 3232 |

Weapons goals 1/16→1/16, kills13→14; SSG goals8/16→8/16, kills28→29.
Jail1 remains0/8 goals. Less episode time includes earlier losing outcomes
and is not evidence of faster successful completion. This update is not
promoted: no success-rate improvement was demonstrated.
Update9 accepted five actor steps, final approximate KL0.004999604.
Post-update captures also passed CUDA finalization: 32/32 cases, 3060 rows,
no training on these validation records, in
`combat-group-loadout-quality-after-processing-v1-20261010`.
Audit of after-update intent is stored in
`combat-group-loadout-target-audit-after-v1-20261010.json`.

Next diagnostic: measure learning and KL/entropy separately for target,
continuous aim/movement, attack and weapon outputs on CUDA before changing
the optimizer or adding a target-switch penalty. Shared actor updates and
a global KL limit may constrain categorical target learning, but this is
currently a hypothesis, not a measured cause. Also retain paired group
regressions when expanding a two-to-three-enemy curriculum.

Follow-up: [CUDA head audit and target-only experiment](combat_target_head_learning_20261010.md)
measured negligible target change in the joint update and verified an explicit
target-only optimizer fork. Its32-battle held-out result was8 goals versus9
for the parent/joint variants, so it is not promoted. Next: broader staged
group training rather than repeated tuning on the same validation sample.
