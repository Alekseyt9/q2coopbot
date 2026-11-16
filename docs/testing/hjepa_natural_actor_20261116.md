# H-JEPA primitive optimizer scaling experiment

This follows the failed first primitive PPO screen documented in
`hjepa_actor_ppo_20261116.md`. Artifact root:
`workspace/artifacts/hjepa-natural-actor-v1-20261116`.
Live U1 and all prior experimental weights remain unchanged. No promotion.

## Hypothesis and predefined comparison

The first pilot's continuous distribution change was almost entirely aim KL.
Movement raw standard deviation is 0.15 and aim standard deviation 0.003.
One unscaled Adam parameter step can therefore spend its KL budget on aim
while barely changing movement. This is an optimization hypothesis, not
proof of the cause of the previous failed combat gate.

Implement `train_combat_hjepa_actor_natural.py`: optimize the same clipped
native primitive surrogate with an approximate damped natural-gradient
direction, using local mean-KL Hessian-vector products and conjugate
gradients. Encoder0, actor and all four log standard deviations train;
weapon and auxiliary world heads remain frozen. No action factor is frozen
and no new gameplay rule is added. This is not Adam PPO or a claim to
reproduce paper TRPO.

Before fitting, fix four updates, clipping 0.1, behavior mean KL limit 0.02,
Fisher damping 0.01, 40 CG iterations, relative residual target 1e-10 and
line search through 1/4096. Acceptance requires finite values, mean behavior
KL within the limit, positive surrogate gain and at least 10% of the scaled
linear predicted gain. A rejected search restores the previous parameters.
There are no persistent optimizer moments in this solver.

Fresh collection uses the unchanged bridge parent and four primitive seeds
261990–261993, with frozen weapon seed 261700. The planned TRAIN511 cohort
could not be generated: city1 seed1922180511 exhausted 128 actor-overlap
attempts. TRAIN512 is outside the registered cohort. No gameplay occurred
for either; the verified replacement is TRAIN509. Its 16 seeds do not appear
in 70 previous H-JEPA plan files (`freshness.json`). This is a bounded local
freshness check, not a claim about unknown external experiments.

## Native data and fitting

64 actual games on 16 generated TRAIN509 worlds, eight known maps.
Go source replay exports 7906 proposals: 2157, 1767, 1844 and 2138 by stream.
Independent likelihood audits pass for all four; maximum log-probability
error 1.91e-13. Existing native first-life return-to-go admission retains
7890 decisions and masks 16 nonexclusive/post-endpoint choices. Reward,
baseline and features are unchanged from the preceding pilot.

Both Adam and the natural-gradient candidate fit these same admitted rows
for four full-batch updates. The natural solver accepts scales 1, 1/2,
1/4 and 1/4. Final mean KL is 0.0135997 versus Adam 0.00174489.
The achieved KLs differ; equal update count is not equal policy deformation.

| Raw mean change, RMS in behavior sigma | Adam | Natural gradient |
| --- | ---: | ---: |
| Forward | 0.00514 | 0.02294 |
| Side | 0.00899 | 0.03102 |
| Yaw | 0.04843 | 0.12667 |
| Pitch | 0.03218 | 0.09700 |

Movement changes are approximately 4.47×/3.45× larger than Adam, but aim
still contributes most continuous KL. This does not establish that scaling
alone explains any subsequent combat difference.

The 40-iteration CG solves are approximate: relative residuals range from
0.114 to 0.224, so the requested 1e-10 tolerance is not reached. Final maximum
per-observation KL is 1.918, much larger than its mean; the configured gate
is explicitly on the mean. These limits must stay visible in the evidence.

Five independent numeric tests cover SPD solving, invalid curvature,
Gaussian Fisher blocks, standardized movement/aim scaling and parameter
round trips. True 2+2 versus four native-batch CUDA updates exactly match
weights, histories, source bindings and model JSON bytes. Updated Go actor
parity passes 176 values for each optimizer: max error 3.56e-15 (Adam) and
1.78e-15 (natural). Frozen nonprimitive components are checked separately.

## Validation screen

Protocol fixed before fitting: validation445, 16 fresh worlds on the same
eight known maps, with parent/Adam/natural primitive-mean and one coupled
sampled mode, frozen U1, and rules. 128 actual games, not 128 independent
worlds. No overlapping validation445 seed appears in the same 70 prior
H-JEPA plans. Reserved442 remains unplayed; viewed444 is not training data.

Learned models use `stock_weapons_fire_v1`; rules uses its supported
`legacy_v1` contract in a separate matrix. Comparisons to rules include
dispatch/guard differences. Primitive-mean variants still have the frozen
stochastic weapon head. All variants see identical generated fixtures.

Predefined pilot criteria require, in both mean and sampled modes: at least
two additional natural-gradient wins versus parent, wins no lower than Adam,
and deaths/received HP/self HP no higher than either. A positive screen
would still require independent replication and would not justify automatic
promotion or an autonomous-combat superiority claim.

The authoritative completed result is `native-screen.json`, generated by
`scripts/report_combat_hjepa_natural.py` after both matrices finish.

### Completed result: gate failed

| Variant | Wins | Deaths | Timeouts | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent, primitive means | 2 | 9 | 5 | 967 | 63 |
| Adam, primitive means | 3 | 8 | 5 | 956 | 103 |
| Natural, primitive means | 3 | 9 | 4 | 1011 | 147 |
| Parent, sampled | 3 | 7 | 6 | 865 | 116 |
| Adam, sampled | 3 | 8 | 5 | 865 | 132 |
| Natural, sampled | 2 | 8 | 6 | 896 | 132 |
| Frozen U1 | 6 | 6 | 4 | 773 | 140 |
| Rules | 8 | 5 | 3 | 550 | 80 |

All 128 games completed normally; no capture recovery was needed. The larger
natural-gradient update did not establish combat improvement and is rejected
for promotion. Surrogate gain and stronger movement changes are insufficient
evidence of useful learning. Validation445 is now viewed, never training data.

## Read-only failure diagnosis

`paired-regressions.json`, `self-damage-events.json` and
`grenade-hold-audit.json` bind the observed failures. The additional 84 self HP
in primitive-mean evaluation comes from base1 and base3, +42 HP each, in native
damage events with `mod=24`. Current engine source
`F:/src/quake2/yquake2/src/game/header/local.h:484` defines this as
`MOD_HELD_GRENADE`. These are held hand-grenade explosions, not BFG blasts.

Health-loss capping and survival time matter: parent base1 dies at frame162
before the fuse ends; natural survives to frame192 and self-kills holding the
grenade. Parent base3 self-kills at278 with only18 HP left; natural receives
60 self HP at208 and survives to the timeout at398. Therefore the +84 self HP
does not by itself prove worse survival or explain the aggregate win failure.
It does expose an unlearned fire-release behavior in both source and candidate.

The input already contains observed `GunFrame/30`
(`internal/policy/features.go:235`) and the equipped-weapon context in the871
features. The issue is not an established missing-state bug. On all7906
TRAIN509 proposals (including16masked tails), there are537 hand-grenade
gunframe11 decisions and only1 sampled release. Existing temperature0.25
implies just0.209 expected releases on those recorded states. A read-only
temperature1 calculation gives27.87 expected releases; it is a counterfactual
on fixed observations, not a native rollout or demonstrated improvement.
Other-frame releases are9/7369. Source audit:
`train-fire-exploration-audit.json`.

Next test should increase categorical exploration using declared distribution
parameters on fresh TRAIN data, followed by source-matched learning/evaluation.
Do not train on these validation traces, add a hardcoded release action, reuse
diagnostic interventions as labels, or repeat the failed natural update as a
quality success. A global temperature also changes pose exploration; record
that confound explicitly. Further policy learning must get native evidence
for both firing and releasing instead of assuming a larger movement update
will solve combat.
