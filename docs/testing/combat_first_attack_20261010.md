# First attack after native combat release

Measured with `scripts/report_combat_first_attack.py` on the sealed
`workspace/artifacts/movement-ppo-eval-v2-20261010` evaluation:480 fights,
80 per variant. Inputs are pinned member report hashes, native ordered command
windows, first-life alive steps and the server combat-release marker. The
release is logical server frame98 (native game frame100); preparation is
excluded. No neural execution is involved.

`first-attack.json` retains episode details and hashes of steps/server logs.
Delays below are native server frames, not accelerated wall-clock seconds.

| Variant | Initial Blaster fights with actual launch | Blaster launch mean/max delay | Initial Machinegun ammo decrement observed | Machinegun proxy mean/max delay |
| --- | ---: | ---: | ---: | ---: |
| Instant PPO1 | 40/40 | 0/0 | 37/40 | 1.297/13 |
| Instant PPO2 | 40/40 | 0/0 | 37/40 | 1.811/13 |
| Postmove PPO1 | 40/40 | 0/0 | 40/40 | 0.525/12 |
| Postmove PPO2 | 40/40 | 0/0 | 40/40 | 0.525/12 |
| FireBC | 40/40 | 0/0 | 40/40 | 1.975/17 |
| Rules | 39/40 | 0.564/11 | 33/40 | 1.152/11 |

Blaster launch is native mod1 projectile telemetry joined to an exact exclusive
first-life command interval. Machinegun results use a one-round ammo decrease
between alive first-life observations with the same observed weapon; they are
an observation proxy, not native per-shot proof. Missing values remain unknown,
never zero delay. Means use only known events. Aggregate launch statistics also
include later weapon changes, so use the initial-weapon strata for startup.

All learned variants requested attack by the first or second command after
release. Postmove PPO2 sent attack in all80 fights, mean0.2625frames, median0,
maximum12; one fight exceeded five frames. In that fight, seed1095029,
case9, initial Machinegun, requested attacks on frames98,99,100,102 were
blocked by `barrel_blast_risk`. On frames101 and103..109 the actor requested
no attack. First sent attack and first ammo decrement occurred on frame110.
Thus the12-frame delay includes both a guard intervention and subsequent
policy decisions; it cannot all be assigned to the guard.

This closes the suspicion of systematic first-shot hesitation **after release
for the initial Blaster cases in this corpus**. It does not verify preparation
duration in the previously viewed demo, every map/weapon, or incoming damage
during any preparation prefix.

The same reporter is queued against the retained process handle of
`miss-reward-ab-eval-20261010`, and will run only after that evaluation exits
and its complete member proof exists. This will expose any new hesitation
introduced by the miss penalty alongside gameplay quality.
