# H-JEPA world training coverage

The source-bound audit is `scripts/audit_combat_hjepa_world_coverage.py`.
Its report is `workspace/artifacts/hjepa-world-utility-v1-20261116/coverage.json`.
It verifies retained exporter rows, source digests, native receipt split and all
first-life reward ledgers. Provider counts below exclude raw rows omitted by the
exporter and precede contiguous16-frame clip filtering.

TRAIN437 comprises64 episodes but only16 distinct generated worlds on eight maps,
repeated with four weapon sampling streams. These are not64 different encounters.
There are7685 retained provider transitions,3414 with guard interventions and724
with nonzero first-life native reward. Observed equipped-weapon coverage is highly
uneven: Blaster3929 rows, BFG80, Rocket Launcher65, Grenade Launcher62. A weapon
being available in a fixture is not equivalent to having executed its dynamics.

VALIDATION440 contains16 different worlds in32 episodes and3856 provider rows.
Its native rewards also reconcile exactly. Every observed weapon type appears
in TRAIN at least briefly, but coverage and initial-weapon proportions differ.
These findings identify limited support; they do not prove it is the sole cause
of poor forecasts or establish a particular sample count as sufficient.

## New collection

`workspace/artifacts/hjepa-world-coverage-v1-20261116` freezes a new64-world TRAIN
plan on the same16 registered descriptors/eight maps, offsets460..463. No worlds
overlap the prior TRAIN437 cohort. Source behavior is curriculum round3 with only
weapon sampling seed changed to261900. No outcome-based acceptance, equipment
filtering, live-model changes or validation/test training is allowed.

The generator yields36 Blaster initial worlds and28 with other initial weapons,
including all ten other weapon types. This remains the generator distribution,
not a balanced-equipment experiment. Observed equipped coverage must be measured
after capture; no minimum coverage or improvement is assumed in advance.

The internal split is fixed before inspecting results: each descriptor's first
three generated instances form48 fit worlds, its fourth instance forms16 internal
holdout worlds. Both retain their registered TRAIN split. Entire world identities
stay together; sampling streams must not cross the partition. This internal
holdout does not replace fresh external validation. Membership and the collection
protocol are saved alongside the plan. Model fitting has not started.

## Collection completed

All64 native episodes finished with complete provenance receipts. Go exported
9119 world transitions;6670 are retained provider transitions,2913 guarded.
The native ledgers reconcile for all64 first lives. There are1104 admitted
provider transitions with nonzero reward, compared with724 in the earlier batch.
The generator spans14 monster classes, including tank commander in two worlds.

Observed equipped coverage includes BFG154, Rocket Launcher188, Machinegun917,
Shotgun521 and Grenade Launcher45 provider transitions. Thus world diversity and
some equipment coverage increased, while grenade-launcher support remains sparse.
Counts alone do not establish prediction accuracy or improvement in combat.

All new captured weapon choices passed the inventory audit:663 non-keep requests,
294 different from the equipped weapon,101 observed changes,75 temporally after
a matching request, zero unavailable choices. Engine autoswitch is not excluded.
Weapon audit and world-export counts differ because the exporter retains only
eligible contiguous native next-state transitions, not every provider proposal.

Frozen Go replay also passed:7383 provider proposals, zero mismatches. The extra
14 proposals beyond the weapon audit are nonexclusive terminal windows, which
that audit omits explicitly. All collection and replay processes have completed.
