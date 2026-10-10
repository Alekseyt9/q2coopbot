# Combat movement at walls, 2026-10-10

The recorded PPO loss seed 730024 fires while staying exposed. During the
first life the policy requested movement, but guards stopped 50 of 64 commands:
42 `static_hull_blocked` and 8 `unsupported_motion_guard`. The successful seed
730027 also stopped 59 of 72 requested movements. These are command counters,
not proof that every stopped command would have produced useful evasion.

The ground guard now tries retaining only forward or only side from the policy's
original command after a static hull rejection. It prefers the smaller action
change and recomputes displacement including current momentum. Each candidate
must pass static collision/support, dynamic door and static laser checks. It
does not reverse directions, add acceleration, aim, fire or invent a route.
Jump, duck, airborne, missing support and other hazard cases retain their prior
restrictions. This change therefore does not yet enable airborne dodging.

Validation: a regression test reproduces the native frame-115 wall position,
preserves the requested strafe and aim/fire, rejects unsupported motion and
residual momentum toward the wall, and rejects head-on movement without adding
a strafe. Four focused non-neural guard/ownership tests passed with real base1
BSP. Neural inference tests and training were not run on CPU.

Native probe: 16 independent runs, 16 pool slots, timescale 2, frozen Postmove PPO
update1 weights. Four existing solo/mixed Blaster/Machinegun recipes use the
exact same generated validation fixtures and seeds 24..27 as the old evaluation.
The fresh verified plan has new source bindings and exact fixture equality.
This is a guard change evaluation, not training or a promotion of the model.
The baseline is the recorded prior capture; native reruns are not guaranteed
fully deterministic. Results are sealed under
`workspace/artifacts/combat-movement-component-v1-20261010`.

## Native results

All 16 captures passed the existing strict member verifier. The pool completed
with unchanged sources and zero frame gaps. Exact weights and generated
fixtures were preserved; no neural training occurred in this probe.

| Measurement | Previous guard | Component clipping |
| --- | ---: | ---: |
| Wins | 8/16 | 8/16 |
| First-life deaths | 7/16 | 7/16 |
| Mean received health damage | 63.69 | 67.25 |
| Alive command frames with planar speed below 5 | 1145/1570 (72.93%) | 1003/1640 (61.16%) |
| Requested movement commands fully stopped | 1048/1306 (80.25%) | 984/1486 (66.22%) |

The displayed loss seed 730024 remains a loss, but stopped movement falls from
50/64 to 21/74 and almost stationary frames from 29/64 to 12/74. The earlier
winning seed 730027 now loses; seed 730026 becomes a win. This is evidence of
improved movement execution, not improved combat quality or model superiority.
Pooled frame fractions are descriptive and the trajectory lengths differ.

`movement-comparison.json` retains per-seed first-life counters and hashes of
both source reports and datasets. `recovery/verified-members.json` and
`quality-report.json` retain the existing capture and quality proofs.

Next learning work must collect fresh own-policy trajectories with this guard
and train on CUDA. Old trajectories have a different execution environment and
must not be represented as fresh on-policy rollouts. Prioritize wall escape,
changing position under aimed attacks, and movement/aim coupling; additional
airborne motion requires a checked trajectory implementation. Keep validation
separate and do not infer a gain from a single improved movement example.
